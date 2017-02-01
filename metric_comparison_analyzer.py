#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import time
import re
import bisect
from collections import defaultdict
from tqdm import tqdm

from data_util.data_reader import ProjectInfo
from data_util.data_reader import IterationGrading
from pivotal_tracker_api.tracker_api import TrackerApi

class MetricComparisonAnalyzer(object):
  """
    Compare metrics with grading.
  """
  def __init__(self, tokens, project_info, iteration_grading):
    super(MetricComparisonAnalyzer, self).__init__()
    self.pt_client = TrackerApi(tokens['pivotal_tracker']['token'])
    self.project_info = project_info
    self.iteration_grading = iteration_grading
    self.out_header = 'metric_comparison'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)
    self.ROOT_PATH = '/Users/Joe/Projects/TeamScope/analysis'

  def iteration_metrics(self, proj):
    """
      Get Tracker metrics for each iteration for a given project.
      This function stems from resample in MetricTracker class. Changes are made for analysis purpose.
    """
    stories = self.pt_client.get_stories(proj['tracker'])
    with open('{}/conf/iterations.json'.format(self.ROOT_PATH), 'r') as f_in:
      iterations = json.load(f_in)
    iterations = [time.mktime(time.strptime(x, '%Y-%m-%d')) for x in iterations]
    iteration_data = defaultdict(lambda: defaultdict(lambda: []))
    for story in stories:
      # Count number of different states
      if not story['current_state'] in ['accepted', 'delivered']:
        continue
      
      stime = time.mktime(time.strptime(story['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
      transitions = self.pt_client.get_story_transitions(proj['tracker'], story['id'])
      state_time_owner = {}
      for trans in transitions:
        occurred_at = time.mktime(time.strptime(trans['occurred_at'], '%Y-%m-%dT%H:%M:%SZ'))
        performed_by = trans['performed_by_id']
        state_time_owner[trans['state']] = (occurred_at, performed_by)

      # Calculate velocities
      if 'started' in state_time_owner:
        stime = state_time_owner['started'][0]
      etime = time.mktime(time.strptime(story['updated_at'], '%Y-%m-%dT%H:%M:%SZ'))
      if 'finished' in state_time_owner:
        etime = state_time_owner['finished'][0]
      dtime = etime
      if 'delivered' in state_time_owner:
        dtime = state_time_owner['delivered'][0]
      atime = dtime
      if 'accepted' in state_time_owner:
        atime = state_time_owner['accepted'][0]

      niter = bisect.bisect(iterations, stime)
      story_points = story['estimate'] if 'estimate' in story and story['estimate'] != 0 else 1

      iteration_data[niter]['velocity'].append(story_points)
      iteration_data[niter]['time'].append((story_points, etime - stime))
      iteration_data[niter]['review'].append(dtime - etime)
      iteration_data[niter]['customer'].append(atime - dtime)
    return iteration_data

  def get_grades(self, proj):
    pid = proj['ID']
    grades = {}
    for item in self.iteration_grading:
      if item['Team ID'] != pid:
        continue
      if item['phase'] != 2:
        continue
      grades[item['iteration']] = item['Pivotal Tracker (5)']
    return grades

  def comparison(self, proj):
    data = self.iteration_metrics(proj)
    grad = self.get_grades(proj)
    features, values = [], []
    for i in range(4):
      features.append(self._extract(data[i+1]))
      if grad[i+1]:
        values.append(float(grad[i+1]))
      else:
        values.append(0)
    return features, values

  def correlation(self):
    features, values = [], []
    for proj in tqdm(self.project_info):
      tmp_feature, tmp_val = self.comparison(proj)
      features.extend(tmp_feature)
      values.extend(tmp_val)
      time.sleep(1)

    plotdata = pd.DataFrame({
      'velocity': [x[0] for x in features],
      'average review time': [np.log(x[1]+1) for x in features],
      'std point estimation': [x[3] for x in features],
      'grade': values})
    fig, ax = plt.subplots()
    sns.pairplot(plotdata)
    plt.savefig('{}/correlation.png'.format(self.out_header))
    plt.close(fig)

  def _extract(self, pt_info):
    velocity = np.sum(pt_info['velocity'])
    avg_reveiw_time = np.average(pt_info['review']) if len(pt_info['review']) > 0 else 0.0
    avg_customer_timer = np.average(pt_info['customer']) if len(pt_info['customer']) > 0 else 0.0
    std_pnt_estimation = np.std([x[1]/x[0] for x in pt_info['time']])
    return np.array([velocity, avg_reveiw_time, avg_customer_timer, std_pnt_estimation])

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  iteration_grading = IterationGrading('data/', 'detailed')
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  analyzer = MetricComparisonAnalyzer(tokens, project_info, iteration_grading)
  # analyzer.comparison(project_info[0])
  analyzer.correlation()
  # analyzer.trend_reload()
  # analyzer.test_analyze_log(project_info[0])

if __name__ == '__main__':
  main()