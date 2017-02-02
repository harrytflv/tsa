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
from metrics.metric_tracker import MetricTracker
from metrics.metric_github import MetricGithub

class MetricComparisonAnalyzer(object):
  """
    Compare metrics with grading.
  """
  def __init__(self, tokens, project_info, iteration_grading):
    super(MetricComparisonAnalyzer, self).__init__()
    self.tokens = tokens
    self.project_info = project_info
    self.iteration_grading = iteration_grading
    self.out_header = 'metric_comparison'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)
    self.ROOT_PATH = '/Users/Joe/Projects/TeamScope/analysis'
    self.metric_list = [MetricTracker, MetricGithub]

  def get_grades(self, proj):
    pid = proj['ID']
    grades = defaultdict(lambda: defaultdict(lambda: 0))
    self.grade_names = ['tracker', 'codeclimate', 'coverage', 'customer', 'total']
    for item in self.iteration_grading:
      if item['Team ID'] != pid:
        continue
      if item['phase'] == 2:
        grades[item['iteration']]['tracker'] = float(item['Pivotal Tracker (5)']) if item['Pivotal Tracker (5)'] else None
        grades[item['iteration']]['total'] = float(item['Total']) if item['Total'] else None
        grades[item['iteration']]['codeclimate'] = float(item['Code Climate (3)']) if not item['Code Climate (3)'] in ['', 'Not working'] else None
        grades[item['iteration']]['coverage'] = float(item['Test Coverage (3)']) if not item['Test Coverage (3)'] in ['', 'coverage unknown', 'Coverage not working'] else None
        grades[item['iteration']]['customer'] = float(item['Customer Communication/Preparedness (4)']) if item['Customer Communication/Preparedness (4)'] else None
    return grades

  def comparison(self, proj):
    metrics = [metric(proj, self.tokens) for metric in self.metric_list]
    data = [metric.metrics() for metric in metrics]
    data_name = [metric.metric_name() for metric in metrics]
    grad = self.get_grades(proj)

    features, labels, values = [], [], []
    for i in range(4):
      tmp_feature = []
      values.append(grad[i+1])
      for item in data:
        tmp_feature.extend(item[i+1])
      features.append(tmp_feature)
    for label in data_name:
      labels.extend(label)
    self.metric_labels = labels
    return features, values

  def correlation(self):
    dataset = defaultdict(lambda: [])
    for proj in tqdm(self.project_info):
      tmp_feature, tmp_val = self.comparison(proj)
      for index, label in enumerate(self.metric_labels):
        dataset[label].extend([feature[index] for feature in tmp_feature])
      for grade_name in self.grade_names:
        dataset[grade_name].extend([val[grade_name] for val in tmp_val])
      time.sleep(1)

    plotdata = pd.DataFrame(dataset)
    fig, ax = plt.subplots()
    sns.pairplot(plotdata, x_vars=self.grade_names, y_vars=self.metric_labels)
    plt.savefig('{}/correlation.png'.format(self.out_header))
    plt.close(fig)

    corr = plotdata.corr(method='pearson')
    corr = corr.loc[self.metric_labels, self.grade_names]
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, linewidths=.5)
    plt.savefig('{}/corr_heatmap.png'.format(self.out_header))
    plt.close(fig)

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  iteration_grading = IterationGrading('data/', 'detailed')
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  analyzer = MetricComparisonAnalyzer(tokens, project_info, iteration_grading)
  # analyzer.comparison(project_info[0])
  analyzer.correlation()

if __name__ == '__main__':
  main()