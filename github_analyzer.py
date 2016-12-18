#!/usr/bin/env 

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import nltk

from github_api.github_api import GithubApi
from data_util.data_reader import ProjectInfo

class GithubAnalyzer(object):
  """docstring for GithubAnalyzer"""
  def __init__(self, token, project_info):
    super(GithubAnalyzer, self).__init__()
    self.client = GithubApi(token)
    self.projects = project_info
    self.commit_cache = False

  def commits(self, reload=False):
    if reload:
      with open('cache/commits.json', 'r') as f_in:
        return json.load(f_in)
    lstCommits = []
    for project in self.projects:
      owner, repo = project['repo']['owner'], project['repo']['repo']
      commits = self.client.get_commits(owner, repo)
      lstCommits.append(commits)
    with open('cache/commits.json', 'w') as f_out:
      json.dump(lstCommits, f_out, sort_keys=True, indent=4, separators=(',', ': '))
    return lstCommits

  def user_commits(self, reload=True):
    all_commits = self.commits(reload)
    with open('cache/new_user_mapping.json', 'r') as f_in:
      user_map = json.load(f_in)
    user_commits = defaultdict(lambda: [])
    for repo in all_commits:
      for commit in repo:
        user = commit['commit']['author']['name']
        if user in user_map:
          user_commits[user_map[user]].append(commit)
    return user_commits

  def get_commit(self, sha):
    if not self.commit_cache:
      with open('cache/sha2commit_new.json', 'r') as f_in:
        self.commit_cache = json.load(f_in)
    if sha in self.commit_cache:
      return self.commit_cache[sha]
    else:
      return False

  ''' 
      Cache all commits. This will take about two hours for cs169 fall 2016. 
      Fix is used for fixing some error messages.
  '''
  def cache_commits(self, fix=True):
    import time
    if fix:
      print('Fix mode')
      with open('cache/sha2commit.json', 'r') as f_in:
        sha2commit = json.load(f_in)
      all_commits = self.commits(reload=True)
      for index, commits in enumerate(all_commits):
        owner, repo = self.projects[index]['repo']['owner'], self.projects[index]['repo']['repo']
        for commit in commits:
          info = sha2commit[commit['sha']]
          if 'message' in info:
            sha2commit[commit['sha']] = self.client.get_commit(owner, repo, commit['sha'])
            time.sleep(0.1)
      with open('cache/sha2commit_new.json', 'w') as f_out:
        json.dump(sha2commit, f_out)
      return
    print('Cache mode')
    dictSha2Commit = {}
    all_commits = self.commits(reload=True)
    for index, commits in enumerate(all_commits):
      owner, repo = self.projects[index]['repo']['owner'], self.projects[index]['repo']['repo']
      for commit in commits:
        dictSha2Commit[commit['sha']] = self.client.get_commit(owner, repo, commit['sha'])
        time.sleep(0.01)
    with open('cache/sha2commit.json', 'w') as f_out:
      json.dump(dictSha2Commit, f_out, sort_keys=True, indent=4, separators=(',', ': '))

  def commits_plot(self):
    lstCommits = self.commits(reload=True)

    plotdata = pd.DataFrame({'x': np.arange(len(lstCommits)), 'y': [len(item) for item in lstCommits]})
    fig, ax = plt.subplots()
    sns.barplot('x', 'y', data=plotdata)
    plt.savefig('results/hist_num_commits.png')
    plt.close(fig)

  def commmits_per_student_plot(self):
    lstCommits = self.commits(reload=True)
    dictStu2Commits = defaultdict(lambda: 0)
    for proj_commit in lstCommits:
      for commit in proj_commit:
        dictStu2Commits[commit['commit']['author']['name']] += 1
    fig, ax = plt.subplots()
    sns.distplot([v for _, v in dictStu2Commits.items()])
    plt.savefig('results/num_commits_per_student.png')
    plt.close(fig)

  def generate_user_map(self):
    with open('cache/user_mapping.json', 'r') as f_in:
      user_map = json.load(f_in)
    self.student_list = user_map.keys()
    lstCommits = self.commits(reload=True)
    setStudents = set()
    for proj_commit in lstCommits:
      for commit in proj_commit:
        setStudents.add(commit['commit']['author']['name'])

    counter = 0
    for student in setStudents:
      if student in self.student_list:
        continue
      choices = self._nearest_neighbor(student)
      for choice in choices:
        inpt = input('{} and {}?'.format(choice, student))
        if inpt == 'Y':
          counter += 1
          user_map[student] = user_map[choice]
          break
        if inpt == 'S':
          break
    print('{}/{}'.format(counter, len(setStudents)))
    with open('cache/new_user_mapping.json', 'w') as f_out:
      json.dump(user_map, f_out)

  def _nearest_neighbor(self, wd):
    choices = list(sorted(self.student_list, key=lambda x: self._distance(x, wd)))[:3]
    return choices

  def _distance(self, wd_1, wd_2):
    wd_2 = ', '.join(reversed(wd_2.split(' ')))
    return nltk.edit_distance(wd_1.lower(), wd_2.lower())

  def extract_proj(self, commit):
    info = commit['url'].split('/')
    ind = info.index('api.github.com')
    return '{}/{}'.format(info[ind+2], info[ind+3])

def main():
  with open('conf/tokens.json', 'r') as f_in:
    token = json.load(f_in)
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  analyzer = GithubAnalyzer(token['github']['token'], project_info)
  # analyzer.commits_plot()
  # analyzer.commmits_per_student_plot()

if __name__ == '__main__':
  main()