#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import json
import os

from github_analyzer import GithubAnalyzer
from github_api.github_api import GithubApi
from pivotal_tracker_api.tracker_api import TrackerApi
from data_util.data_reader import ProjectInfo

class ProcessSegmentAnalyzer(object):
  """
    Segment git commits and correlates them with user stories.
  """
  def __init__(self, project_info):
    """
      Tokens are read from conf/tokens.json file.
      
      Input
        - project_info: dataset contains all course projects
    """
    super(ProcessSegmentAnalyzer, self).__init__()
    self.project_info = project_info
    with open('conf/tokens.json', 'r') as f_in:
      self.token = json.load(f_in)
    self.gt_analyzer = GithubAnalyzer(self.token['github']['token'], self.project_info)
    self.gt_client = GithubApi(self.token['github']['token'])
    self.pt_client = TrackerApi(self.token['pivotal_tracker']['token'])
    self.out_header = 'process_segment'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)

  def correlation(self, proj):
    """
      Generate segmentation for a single project.

      Input
        - proj: a data point in project_info
    """
    pass

  def time_sequence(self, proj):
    """
      Extract time information and files information from commits.

      Input
        - proj: the project
      Output
        - time_sequence: a list of datetime objects
        - file_sequence: a list of file indexes
        - file_dict: a dictionary from file index to file name
    """
    commits = self.gt_client.get_commits(proj['repo']['owner'], proj['repo']['repo'])
    stories = self.pt_client.get_stories(proj['tracker'])

    file_indexer = {}
    time_sequence, file_sequence = [], []
    for cmit in commits:
      # tmp_time = datetime.datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ')
      
      tmp_file_vec = []
      commit = self.gt_analyzer.get_commit(cmit['sha'])
      if not commit:
        print('Commit not found: {}'.format(cmit['sha']))
        continue
      if 'merge' in commit['commit']['message']:
        continue
      for f in commit['files']:
        if not f['filename'] in file_indexer:
          file_indexer[f['filename']] = len(file_indexer)
        tmp_file_vec.append(file_indexer[f['filename']])
      file_sequence.append(tmp_file_vec)
      time_sequence.append(datetime.datetime.strptime(cmit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ'))
    return time_sequence, file_sequence

  def story_time(self, proj):
    """
      Extract time informaiton and story information from pivotral tracker

      Input
        - proj: the project
      Output
    """
    times, info = [], []
    for story in self.pt_client.get_stories(proj['tracker']):
      s = datetime.datetime.strptime(story['created_at'], '%Y-%m-%dT%H:%M:%SZ')
      e = datetime.datetime.strptime(story['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
      times.append((s, e))
      info.append(story)
    return times, info

  def story_time_overlaps(self):
    """
      Plot
        - a counting plot of 'active' user stories over time
    """
    import time
    if not 'story_time' in os.listdir(self.out_header):
      os.mkdir('{}/story_time'.format(self.out_header))
    for proj in self.project_info[1:2]:
      times, info = self.story_time(proj)
      time_to_val = {}
      for s_t, e_t in times:
        time_to_val[s_t] = 1
        time_to_val[e_t] = -1
      time_seq, count_seq = [], []
      counter = 0
      for t in sorted(time_to_val.keys()):
        time_seq.append(t)
        counter += time_to_val[t]
        count_seq.append(counter)

      fig, ax = plt.subplots()
      plt.plot([time.mktime(t.timetuple()) for t in time_seq], count_seq)
      plt.savefig('{}/story_time/{}_{}'.format(self.out_header, proj['ID'], proj['project'].replace(" ", "")))
      plt.close(fig)

  def git_commit_overlaps(self):
    """
      Plot
        - a scatter plot between time and files edited for a given project.
    """
    import time
    if not 'commit_time' in os.listdir(self.out_header):
      os.mkdir('{}/commit_time'.format(self.out_header))
    for proj in self.project_info[1:2]:
      times, files = self.time_sequence(proj)
      sorted_time = sorted(times)
      t_seq, f_seq = [], []
      for i in range(len(times)):
        for f in files[i]:
          # t_seq.append(sorted_time.index(times[i]))
          t_seq.append(time.mktime(times[i].timetuple()))
          f_seq.append(f)
      plotdata = pd.DataFrame({'time':t_seq, 'file':f_seq})

      fig, ax = plt.subplots()
      sns.jointplot(x='time', y='file', data=plotdata)
      plt.savefig('{}/commit_time/{}_{}.png'.format(self.out_header, proj['ID'], proj['project'].replace(" ", "")))
      plt.close(fig)

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  analyzer = ProcessSegmentAnalyzer(project_info)
  # analyzer.git_commit_overlaps()
  analyzer.story_time_overlaps()

if __name__ == '__main__':
  main()