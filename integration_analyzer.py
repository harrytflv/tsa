#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import time
from tqdm import tqdm
import re
from collections import defaultdict

from data_util.data_reader import ProjectInfo
from travis_api.travis_api import TravisApi

class IntegrationAnalyzer(object):
  """
    Analyzer based on TravisCI and GitHub.
  """
  def __init__(self, tokens, project_info):
    super(IntegrationAnalyzer, self).__init__()
    self.tv_client = TravisApi(tokens['travis']['token'])
    self.project_info = project_info
    self.out_header = 'integration'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)

  def builds(self, proj, reload=True):
    """
      Get all builds of the given project.
      Try reload if reload=True. Cache file is cache/builds.json.
    """
    owner, repo = proj['repo']['owner'], proj['repo']['repo']
    key = '{}|{}'.format(owner, repo)
    if reload:
      with open('cache/builds.json', 'r') as f_in:
        build_cache = json.load(f_in)
      if key in build_cache:
        return build_cache[key]
      else:
        builds = self.tv_client.list_builds(owner, repo)
        build_cache[key] = builds
        with open('cache/builds.json', 'w') as f_out:
          json.dump(build_cache, f_out)
        return builds
    else:
      builds = self.tv_client.list_builds(owner, repo)
      build_cache = {key: builds}
      with open('cache/builds.json', 'w') as f_out:
        json.dump(build_cache, f_out)
      return builds

  def trend(self, proj):
    builds = self.builds(proj)
    builds = list(filter(lambda x: x['started_at'] and x['job_ids'], builds))
    builds = sorted(builds, key=lambda bd: bd['started_at'])
    timestamps, status = [], []
    rspec_total, rspec_passed, rspec_coverage = [], [], []
    cucumber_total, cucumber_passed, cucumber_coverage = [], [], []
    for bd in tqdm(builds):
      log_info = {}
      for job in bd['job_ids']:
        log_info.update(self._analyze_log(self.tv_client.get_log(job)))
      if 'cucumber' in log_info and len(log_info['cucumber'])==3:
        cucumber_total.append(log_info['cucumber']['total'])
        cucumber_passed.append(log_info['cucumber']['passed'])
        cucumber_coverage.append(log_info['cucumber']['coverage'])
      else:
        cucumber_total.append(-1)
        cucumber_passed.append(-1)
        cucumber_coverage.append(-1)
      if 'rspec' in log_info and len(log_info['rspec'])==3:
        rspec_total.append(log_info['rspec']['total'])
        rspec_passed.append(log_info['rspec']['passed'])
        rspec_coverage.append(log_info['rspec']['coverage'])
      else:
        rspec_total.append(-1)
        rspec_passed.append(-1)
        rspec_coverage.append(-1)
      timestamps.append(int(time.mktime(time.strptime(bd['started_at'], '%Y-%m-%dT%H:%M:%SZ'))))
      status.append(bd['state'])
    """
    fig, ax1 = plt.subplots()
    ax1.plot(timestamps, cucumber_total, '-', label='Total')
    ax1.plot(timestamps, cucumber_passed, '-', label='Passed')
    ax1.set_ylabel('Number of steps')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(timestamps, cucumber_coverage, '-r', label='Coverage')
    ax2.set_ylabel('Percentage')
    ax2.legend(loc=0)
    plt.savefig('{}/cucumber.png'.format(self.out_header))
    plt.close(fig)

    fig, ax1 = plt.subplots()
    ax1.plot(timestamps, rspec_total, '-', label='Total')
    ax1.plot(timestamps, rspec_passed, '-', label='Passed')
    ax1.set_ylabel('Number of examples')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(timestamps, rspec_coverage, '-r', label='Coverage')
    ax2.set_ylabel('Percentage')
    ax2.legend(loc=0)
    plt.savefig('{}/rspec.png'.format(self.out_header))
    plt.close(fig)
    """
    percent_passed = np.array(list(filter(lambda x: x != -1, cucumber_passed))) / np.array(list(filter(lambda x: x != -1, cucumber_total)))
    fig, ax = plt.subplots()
    plt.plot(100.0*percent_passed, marker='d', label='Percent Passed')
    plt.plot(list(filter(lambda x: x != -1, cucumber_coverage)), marker='o', label='Coverage')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('{}/{}_cucumber.png'.format(self.out_header, proj['ID']))
    plt.close(fig)

    percent_passed = np.array(list(filter(lambda x: x != -1, rspec_passed))) / np.array(list(filter(lambda x: x != -1, rspec_total)))
    fig, ax = plt.subplots()
    plt.plot(100.0*percent_passed, marker='d', label='Percent Passed')
    plt.plot(list(filter(lambda x: x != -1, rspec_coverage)), marker='o', label='Coverage')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('{}/{}_spec.png'.format(self.out_header, proj['ID']))
    plt.close(fig)

    return {'timstamp':timestamps,
            'status': status,
            'rspec_total':rspec_total,
            'rspec_passed':rspec_passed,
            'rspec_coverage':rspec_coverage,
            'cucumber_total':cucumber_total,
            'cucumber_passed':cucumber_passed,
            'cucumber_coverage':cucumber_coverage}

  def trend_reload(self):
    with open('cache/trend_data.json', 'r') as f_in:
      trend_data = json.load(f_in)
    def filter_data(input_lst):
      return list(filter(lambda x: x != -1, input_lst))

    for proj in self.project_info:
      proj_info = trend_data[proj['ID']]
      c_total = np.array(filter_data(proj_info['cucumber_total']))
      c_passed = np.array(filter_data(proj_info['cucumber_passed']))
      c_coverage = np.array(filter_data(proj_info['cucumber_coverage']))

      r_total = np.array(filter_data(proj_info['rspec_total']))
      r_passed = np.array(filter_data(proj_info['rspec_passed']))
      r_coverage = np.array(filter_data(proj_info['rspec_coverage']))

      fig, ax1 = plt.subplots()
      ax1.plot(100.0*c_passed/c_total, marker='d', label='Percent Passed')
      ax1.plot(c_coverage, marker='o', label='Coverage')
      ax1.set_ylabel('Percentage')
      ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
      ax2 = ax1.twinx()
      ax2.plot(c_total, '-sr', label='Test Cases')
      ax2.set_ylabel('Number of test cases')
      ax2.legend()
      plt.savefig('{}/cucumber_{}.png'.format(self.out_header, proj['ID']))
      plt.close(fig)

      fig, ax1 = plt.subplots()
      ax1.plot(100.0*r_passed/r_total, marker='d', label='Percent Passed')
      ax1.plot(r_coverage, marker='o', label='Coverage')
      ax1.set_ylabel('Percentage')
      ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
      ax2 = ax1.twinx()
      ax2.plot(r_total, '-sr', label='Test Cases')
      ax2.set_ylabel('Number of test cases')
      ax2.legend()
      plt.savefig('{}/rspec_{}.png'.format(self.out_header, proj['ID']))
      plt.close(fig)

  def _analyze_log(self, log_in):
    ansi_escape = re.compile(r'\x1b[^m]*m')
    text = ansi_escape.sub('', log_in)
    log_info = defaultdict(lambda: {})
    lines = iter(text.splitlines())
    try:
      while True:
        line = next(lines)
        info_list = line.split(' ')
        if len(info_list) < 3:
          continue
        if info_list[1] == 'steps':
          log_info['cucumber']['total'] = int(info_list[0])
          log_info['cucumber']['passed'] = int(info_list[2][1:])
        if info_list[0] == 'Coverage' and 'Cucumber' in info_list:
          log_info['cucumber']['coverage'] = float(info_list[-2][1:-2])
        if info_list[1] == 'examples,':
          log_info['rspec']['total'] = int(info_list[0])
          log_info['rspec']['passed'] = int(info_list[2])
        if info_list[0] == 'Coverage' and info_list[1] == '=':
          log_info['rspec']['coverage'] = float(info_list[2][:-2])
    except StopIteration:
      return log_info
    return log_info

  def _cucumber_info(self, lines):
    log_info = {}
    while True:
      line = next(lines)
      if '$' in line:
        return log_info
      info_list = line.split(' ')
      if 'steps' in info_list and 'passed' in info_list:
        log_info['cucumber_total'] = int(info_list[0])
        log_info['cucumber_passed'] = int(info_list[2])
        return log_info
    return log_info

  def _rspec_info(self, lines):
    log_info = {}
    while True:
      line = next(lines)
      if '$' in line:
        return log_info
      info_list = line.split(' ')
      if 'examples,' in info_list:
        log_info['cucumber_total'] = int(info_list[0])
        log_info['cucumber_passed'] = int(info_list[2])
    return log_info

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  analyzer = IntegrationAnalyzer(tokens, project_info)
  # print(len(analyzer.builds(project_info[0])))
  # data_cache = {}
  # for proj in project_info:
  #   print('Processing Project {}'.format(proj['ID']))
  #   data = analyzer.trend(proj)
  #   data_cache[proj['ID']] = data
  # with open('cache/trend_data.json', 'w') as f_out:
  #  json.dump(data_cache, f_out)
  analyzer.trend_reload()

if __name__ == '__main__':
  main()