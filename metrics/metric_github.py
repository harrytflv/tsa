#!/usr/bin/env 

import numpy as np
import json
import os
import time
from bisect import bisect
from tqdm import tqdm
from collections import defaultdict
import nltk

from metrics.basic_metric import BasicMetric
from github_api.github_api import GithubApi
from github_analyzer import GithubAnalyzer
from data_util.data_reader import ProjectInfo

class MetricGithub(BasicMetric):
  """All metrics concerning Github"""
  def __init__(self, proj, tokens, **args):
    super(MetricGithub, self).__init__(proj, tokens)

    self.out_header = 'metric_github'
    if not self.out_header in os.listdir(self.ROOT_PATH+'/results'):
      os.mkdir('{}/results/{}'.format(self.ROOT_PATH, self.out_header))
    self.out_header = '{}/results/{}'.format(self.ROOT_PATH, self.out_header)
    self.projsha_commit, self.projsha_pr = {}, {}

  def _load_connection(self):
    self.client = GithubApi(self.tokens['github']['token'])

  def metrics(self, **args):
    commits = self._commits()
    pull_requests = self._pull_requests()
    with open('{}/conf/iterations.json'.format(self.ROOT_PATH), 'r') as f_in:
      iterations = json.load(f_in)
    iterations = [time.mktime(time.strptime(x, '%Y-%m-%d')) for x in iterations]
    iteration_data = defaultdict(lambda: defaultdict(lambda: []))
    for cmit in commits:
      ctime = time.mktime(time.strptime(cmit['commit']['committer']['date'], '%Y-%m-%dT%H:%M:%SZ'))
      nite = bisect(iterations, ctime)
      if not nite in [1, 2, 3, 4]:
        continue
      sha = cmit['sha']
      cmit_info = self._get_commit(sha)
      iteration_data[nite]['num_files'].append(len(cmit_info['files']))
      iteration_data[nite]['comments'].append(cmit_info['commit']['message'])
    for pr in pull_requests:
      if not pr['merged_at']:
        continue
      ctime = time.mktime(time.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
      mtime = time.mktime(time.strptime(pr['merged_at'], '%Y-%m-%dT%H:%M:%SZ'))
      nite = bisect(iterations, mtime)
      if not nite in [1, 2, 3, 4]:
        continue
      pr_info = self._get_pull_request(pr['number'])
      num_comments = pr_info['comments']

      iteration_data[nite]['review_time'].append(mtime-ctime)
      iteration_data[nite]['pr_comments'].append(num_comments)

    result = defaultdict(lambda: [None for _ in self.metric_name()])
    for k, v in iteration_data.items():
      result[k] = self._extract(v)
    return result

  def metric_name(self):
    return ['Files Edited', 'Message Length', 'PR Review', 'PR Comments']

  def _extract(self, info):
    total_num_files = np.sum(info['num_files'])
    avg_msg_length = np.average([len(nltk.word_tokenize(x)) for x in info['comments']])
    avg_review_time = np.average([np.log(x+1) for x in info['review_time']]) if len(info['review_time']) > 0 else None
    avg_num_comments = np.average(info['pr_comments']) if len(info['pr_comments']) > 0 else None
    return [total_num_files, avg_msg_length, avg_review_time, avg_num_comments]

  def _commits(self, reload=True):
    proj_commit = {}
    if 'proj2commits.json' in os.listdir('{}/cache/'.format(self.ROOT_PATH)):
      with open('{}/cache/proj2commits.json'.format(self.ROOT_PATH), 'r') as f_in:
        proj_commit = json.load(f_in)
      if reload and self.proj['ID'] in proj_commit:
        return proj_commit[self.proj['ID']]
    owner, repo = self.proj['repo']['owner'], self.proj['repo']['repo']
    commits = self.client.get_commits(owner, repo)
    proj_commit[self.proj['ID']] = commits
    with open('{}/cache/proj2commits.json'.format(self.ROOT_PATH), 'w') as f_out:
      json.dump(proj_commit, f_out, sort_keys=True, indent=4, separators=(',', ': '))
    return commits

  def _get_commit(self, sha, reload=True):
    dict_key = '{}:{}'.format(self.proj['ID'], sha)
    if dict_key in self.projsha_commit:
      return self.projsha_commit[dict_key]
    if 'projsha2commit.json' in os.listdir('{}/cache/'.format(self.ROOT_PATH)):
      with open('{}/cache/projsha2commit.json'.format(self.ROOT_PATH), 'r') as f_in:
        self.projsha_commit = json.load(f_in)
      if reload and dict_key in self.projsha_commit:
        return self.projsha_commit[dict_key]
    owner, repo = self.proj['repo']['owner'], self.proj['repo']['repo']
    commit = self.client.get_commit(owner, repo, sha)
    multiplier, sleep_time = 2, 0.1
    while 'message' in commit:
      print(commit['message'])
      time.sleep(sleep_time)
      commit = self.client.get_commit(owner, repo, sha)
      sleep_time *= multiplier
    self.projsha_commit[dict_key] = commit
    with open('{}/cache/projsha2commit.json'.format(self.ROOT_PATH), 'w') as f_out:
      json.dump(self.projsha_commit, f_out, sort_keys=True, indent=4, separators=(',', ': '))
    return commit

  def _pull_requests(self, reload=True):
    proj_requests = {}
    if 'proj2prs.json' in os.listdir('{}/cache/'.format(self.ROOT_PATH)):
      with open('{}/cache/proj2prs.json'.format(self.ROOT_PATH), 'r') as f_in:
        proj_requests = json.load(f_in)
      if reload and self.proj['ID'] in proj_requests:
        return proj_requests[self.proj['ID']]
    owner, repo = self.proj['repo']['owner'], self.proj['repo']['repo']
    prs = self.client.get_pull_requests(owner, repo)
    proj_requests[self.proj['ID']] = prs
    with open('{}/cache/proj2prs.json'.format(self.ROOT_PATH), 'w') as f_out:
      json.dump(proj_requests, f_out, sort_keys=True, indent=4, separators=(',', ': '))
    return prs

  def _get_pull_request(self, number, reload=True):
    dict_key = '{}:{}'.format(self.proj['ID'], number)
    if dict_key in self.projsha_pr:
      return self.projsha_pr[dict_key]
    if 'projsha2pr.json' in os.listdir('{}/cache/'.format(self.ROOT_PATH)):
      with open('{}/cache/projsha2pr.json'.format(self.ROOT_PATH), 'r') as f_in:
        self.projsha_pr = json.load(f_in)
      if reload and dict_key in self.projsha_pr:
        return self.projsha_pr[dict_key]
    owner, repo = self.proj['repo']['owner'], self.proj['repo']['repo']
    pr = self.client.get_pull_request(owner, repo, number)
    multiplier, sleep_time = 2, 1
    while 'message' in pr:
      print(pr['message'])
      time.sleep(sleep_time)
      pr = self.client.get_pull_request(owner, repo, number)
      sleep_time *= multiplier
    self.projsha_pr[dict_key] = pr
    with open('{}/cache/projsha2pr.json'.format(self.ROOT_PATH), 'w') as f_out:
      json.dump(self.projsha_pr, f_out, sort_keys=True, indent=4, separators=(',', ': '))
    return pr

  def dump(self):
    if 'projsha2commit.json' in os.listdir('{}/cache/'.format(self.ROOT_PATH)):
      with open('{}/cache/projsha2commit.json'.format(self.ROOT_PATH), 'r') as f_in:
        tmp_projsha_commit = json.load(f_in)
      self.projsha_commit.update(tmp_projsha_commit)
    with open('{}/cache/projsha2commit.json'.format(self.ROOT_PATH), 'w') as f_out:
      json.dump(self.projsha_commit, f_out, sort_keys=True, indent=4, separators=(',', ': '))

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  for proj in project_info:
    metric = MetricGithub(proj, tokens)
    print(metric.metrics())

if __name__ == '__main__':
  main()