#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import json
import time
import datetime
from tqdm import tqdm

from data_util.data_reader import ProjectInfo
from pivotal_tracker_api.tracker_api import TrackerApi

class PivotalTrackerAnalyzer(object):
  """
    Generate pivotal tracker statistics and visualization.
  """
  def __init__(self, project_info, token):
    """
      Input
        - project_info: project information data
        - token: tracker access token
    """
    super(PivotalTrackerAnalyzer, self).__init__()
    self.project_info = project_info
    self.client = TrackerApi(token)

  def stories(self, proj):
    """
      Get all stories for a project
    """
    return self.client.get_stories(proj['tracker'])
  
  def story_assign(self, reload=False):
    '''
      Generate the task assignment within team. Ownership is defined by own relation.
      If reload=True, use cached file. Otherwise generate and cache the result.

      Output
        - dictPnts: dictionary from a tracker user name to fraction of points
        - dictNum: dictionary from a tracker user name to fraction of number of stories
    '''
    if reload:
      with open('cache/student_points_num.json', 'r') as f_in:
        cache = json.load(f_in)
      return cache['points'], cache['number']
    dictPnts, dictNum = {}, {}
    for project in tqdm(self.project_info, desc='Project'):
      dictStu2Pnts, dictStu2Num = self._story_assign(project['tracker'], len(project['students']))
      dictPnts.update(dictStu2Pnts)
      dictNum.update(dictStu2Num)
    with open('cache/student_points_num.json', 'w') as f_out:
      json.dump({'points': dictPnts, 'number': dictNum}, f_out)
    return dictPnts, dictNum

  def story_assign_plot(self, reload=False):
    '''
      Plot
        - histogram of story assignment, both fraction of points and numbers are plotted
    '''
    dictPnts, dictNums = self.story_assign()
    lstPnts = [v for _, v in dictPnts.items()]
    lstNums = [v for _, v in dictNums.items()]

    fig, ax = plt.subplots()
    sns.distplot(lstPnts, label='Percentage of Points')
    sns.distplot(lstNums, label='Percentage of Total Stories')
    plt.legend()
    plt.savefig('results/story_assign.png')
    plt.close(fig)

  def _story_assign(self, project_id, num_stu):
    '''Process single project.'''
    dictStu2Pnts, dictStu2Num = defaultdict(lambda: 0), defaultdict(lambda: 0)
    total_pnts, total_num = 0, 0
    for story in self.client.get_stories(project_id):
      pnts = story['estimate'] if 'estimate' in story else 1
      total_pnts += pnts
      total_num += 1
      owners = self.client.get_story_owners(project_id, story['id'])
      # owners = story['owner_ids']
      for owner in owners:
        dictStu2Num[owner['name']] += 1
        dictStu2Pnts[owner['name']] += pnts
      time.sleep(0.1) # Prevent too frequent requests
    for i in range(num_stu-len(dictStu2Num)):
      dictStu2Num[project_id+str(i)+'#'] = 0
      dictStu2Pnts[project_id+str(i)+'#'] = 0
    return {k: float(v)/float(total_pnts) for k, v in dictStu2Pnts.items()},\
           {k: float(v)/float(total_num) for k, v in dictStu2Num.items()}

  def iteration_points(self, reload=False):
    '''
      Generate the task assignment within team divided by iterations. Ownership is defined by own relation.
      If reload=True, use cached file. Otherwise generate and cache the result.

      Output
        - dictPnts: dictionary from a tracker user name to a list of fraction of points
        - dictNum: dictionary from a tracker user name to a list of fraction of number of stories
    '''
    if reload:
      with open('cache/student_iter_pnts.json', 'r') as f_in:
        cache = json.load(f_in)
      return cache['points'], cache['number']
    dictPnts, dictNum = {}, {}
    for project in tqdm(self.project_info, desc='Project'):
      dictStu2Pnts, dictStu2Num = self._iteration_points(project['tracker'], len(project['students']))
      dictPnts.update(dictStu2Pnts)
      dictNum.update(dictStu2Num)
    with open('cache/student_iter_pnts.json', 'w') as f_out:
      json.dump({'points': dictPnts, 'number': dictNum}, f_out)
    return dictPnts, dictNum

  def _iteration_points(self, project_id, num_stu):
    """Process single project"""
    dictStu2Pnts, dictStu2Num = defaultdict(lambda: np.zeros((4,))), defaultdict(lambda: np.zeros((4,)))
    total_pnts, total_num = np.ones((4,)), np.ones((4,))
    with open('conf/iterations.json', 'r') as f_in:
      timestamps = json.load(f_in)
    timestamps = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in timestamps]
    for story in self.client.get_stories(project_id):
      # t = datetime.datetime.fromtimestamp(int(story['updated_at'])/1e3)
      t = datetime.datetime.strptime(story['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
      ind = np.searchsorted(timestamps, t)
      pnts = story['estimate'] if 'estimate' in story else 1
  
      if ind in [1, 2, 3, 4]:
        ind = ind - 1
        total_pnts[ind] += pnts
        total_num[ind] += 1
        owners = self.client.get_story_owners(project_id, story['id'])
        for owner in owners:
          dictStu2Num[owner['name']][ind] += 1
          dictStu2Pnts[owner['name']][ind] += pnts
      time.sleep(0.1) # Prevent too frequent requests
    for i in range(num_stu-len(dictStu2Num)):
      dictStu2Num[project_id+str(i)+'#'] = np.zeros((4,))
      dictStu2Pnts[project_id+str(i)+'#'] = np.zeros((4,))
    return {k: (v/total_pnts).tolist() for k, v in dictStu2Pnts.items()},\
           {k: (v/total_num).tolist() for k, v in dictStu2Num.items()}


def main():
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'project-info')
  analyzer = PivotalTrackerAnalyzer(project_info, tokens['pivotal_tracker']['token'])
  # analyzer.story_assign_plot()
  analyzer.iteration_points()

if __name__ == '__main__':
  main()