#!/usr/bin/env 

import sys
import os
import json
import time
import datetime
from collections import defaultdict
import bisect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from metrics.basic_metric import BasicMetric
from pivotal_tracker_api.tracker_api import TrackerApi
from data_util.data_reader import ProjectInfo

class MetricTracker(BasicMetric):
  """All metrics concerning pivotal tracker"""
  def __init__(self, proj, token, **args):
    super(MetricTracker, self).__init__(proj, token)

    self.out_header = 'metric_tracker'
    if not self.out_header in os.listdir(self.ROOT_PATH+'/results'):
      os.mkdir('{}/results/{}'.format(self.ROOT_PATH, self.out_header))
    self.out_header = '{}/results/{}'.format(self.ROOT_PATH, self.out_header)

  def _load_connection(self):
    self.client = TrackerApi(self.tokens['pivotal_tracker']['token'])

  def resample(self):
    stories = self.client.get_stories(self.proj['tracker'])
    with open('{}/conf/iterations.json'.format(self.ROOT_PATH), 'r') as f_in:
      iterations = json.load(f_in)
    iterations = [time.mktime(time.strptime(x, '%Y-%m-%d')) for x in iterations]
    state_count = defaultdict(lambda: 0)
    velocity_count = defaultdict(lambda: 0.0)
    pnts_time = []
    owner_info = []
    for story in stories:
      # Count number of different states
      state_count[story['current_state']] += 1
      if not story['current_state'] in ['accepted', 'delivered']:
        continue

      stime = time.mktime(time.strptime(story['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
      transitions = self.client.get_story_transitions(self.proj['tracker'], story['id'])
      trans_owner_time = []
      for trans in transitions:
        occurred_at = time.mktime(time.strptime(trans['occurred_at'], '%Y-%m-%dT%H:%M:%SZ'))
        if trans['state'] == 'started':
          stime = occurred_at
        trans_owner_time.append((trans['state'], trans['performed_by_id'], occurred_at))

      # Calculate velocities
      etime = time.mktime(time.strptime(story['updated_at'], '%Y-%m-%dT%H:%M:%SZ'))
      niter = bisect.bisect(iterations, etime)
      story_points = story['estimate'] if 'estimate' in story else 1
      velocity_count[niter] += story_points

      # Time versus points
      pnts_time.append((story_points, etime - stime))

      # Ownership
      for tot in trans_owner_time:
        niter = bisect.bisect(iterations, tot[2])
        owner_info.append((tot[0], tot[1], niter))
    self.data = {
      'velocity': velocity_count,
      'state': state_count,
      'pnts_time': pnts_time,
      'ownership': owner_info
    }

    return self.data, datetime.datetime.now()

  def graph(self):
    velocity_count, state_count = self.data['velocity'], self.data['state']
    pnts_time, owner_info = self.data['pnts_time'], self.data['ownership']

    fig, ax = plt.subplots()
    ind = np.arange(len(state_count))
    width = 0.5
    labels = [k for k in state_count]
    ax.bar(ind, [state_count[v] for v in labels], width)
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels(labels)
    plt.savefig('{}/state_{}.png'.format(self.out_header, self.proj['ID']))
    plt.close(fig)

    fig, ax = plt.subplots()
    ind = np.arange(4)
    ax.plot(ind, [velocity_count[i+1] for i in range(4)])
    ax.set_xticks(ind)
    ax.set_xticklabels(['Iter {}'.format(i+1) for i in range(4)])
    plt.savefig('{}/velocity_{}.png'.format(self.out_header, self.proj['ID']))
    plt.close(fig)

    if len(pnts_time) > 0:
      fig, ax = plt.subplots()
      plotdata = pd.DataFrame({'points':[item[0] for item in pnts_time], 'time':[item[1] for item in pnts_time]})
      sns.pointplot(x='points', y='time', data=plotdata)
      plt.savefig('{}/time_pnts_{}.png'.format(self.out_header, self.proj['ID']))
      plt.close(fig)

    if len(owner_info) > 0:
      fig, ax = plt.subplots()
      plotdata = pd.DataFrame({'owner':[item[1] for item in owner_info], 'iteration':[item[2] for item in owner_info]})
      sns.countplot(x='iteration', hue='owner', data=plotdata)
      plt.savefig('{}/ownership_{}.png'.format(self.out_header, self.proj['ID']))
      plt.close(fig)

  def metrics(self, **args):
    stories = self.client.get_stories(self.proj['tracker'])
    with open('{}/conf/iterations.json'.format(self.ROOT_PATH), 'r') as f_in:
      iterations = json.load(f_in)
    iterations = [time.mktime(time.strptime(x, '%Y-%m-%d')) for x in iterations]
    iteration_data = defaultdict(lambda: defaultdict(lambda: []))
    for story in stories:
      # Count number of different states
      if not story['current_state'] in ['accepted', 'delivered']:
        continue
      
      stime = time.mktime(time.strptime(story['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
      transitions = self.client.get_story_transitions(self.proj['tracker'], story['id'])
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

      niter = bisect.bisect(iterations, etime)
      story_points = story['estimate'] if 'estimate' in story and story['estimate'] != 0 else 1

      iteration_data[niter]['velocity'].append(story_points)
      iteration_data[niter]['time'].append((story_points, etime - stime))
      iteration_data[niter]['review'].append(dtime - etime)
      iteration_data[niter]['customer'].append(atime - dtime)
    result = defaultdict(lambda: [None for _ in self.metric_name()])
    for k, v in iteration_data.items():
      result[k] = self._extract(v)
    return result

  def metric_name(self):
    return ['average velocity', 'average review time', 'correlation']

  def _extract(self, pt_info):
    import math
    from scipy.stats import spearmanr
    velocity = np.sum(pt_info['velocity'])
    avg_reveiw_time = np.average([np.log(x+1) for x in pt_info['review']]) if len(pt_info['review']) > 0 else None
    avg_customer_timer = np.average(pt_info['customer']) if len(pt_info['customer']) > 0 else None
    r = spearmanr([x[1] for x in pt_info['time']], [x[0] for x in pt_info['time']])[0]
    r = r if not math.isnan(r) else None
    return [velocity, avg_reveiw_time, r]

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  for proj in tqdm(project_info):
    metric = MetricTracker(proj, token=tokens)
    metric.metrics()

if __name__ == '__main__':
  main()