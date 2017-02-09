#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from collections import defaultdict

from github_api.mysql_api import MysqlApi

class MSRAnalyzer(object):
  """Analyze basic information about MSR14 dataset"""
  def __init__(self, config):
    super(MSRAnalyzer, self).__init__()
    self.client = MysqlApi(config)
    self.out_header = 'metric_tracker'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)

  def fork_correlation(self):
    def get_fork_root(proj):
      proj = proj
      while proj.forked_from:
        proj = id2proj[proj.forked_from]
      return proj.id
    projects = self.client.get_projects()
    id2proj = {proj.id:proj for proj in projects}
    p_count = defaultdict(lambda: 0)
    for proj in projects:
      p_count[get_fork_root(proj)] += 1
    p_ary, f_ary, i_ary = [], [], []
    for p in p_count:
      num_wtch = len(self.client.get_watchers(id2proj[p]))
      if num_wtch > 0:
        p_ary.append(p_count[p])
        f_ary.append(num_wtch)

    plotdata = pd.DataFrame({'forks':p_ary, 'watchers': f_ary})
    fig, ax = plt.subplots()
    sns.jointplot(x='forks', y='watchers', data=plotdata)
    plt.savefig('{}/fork_watch_correlation.png'.format(self.out_header))
    plt.close(fig)

def main():
  with open('conf/mysql.json', 'r') as f_in:
    conf = json.load(f_in)
  analyzer = MSRAnalyzer(conf)
  analyzer.fork_correlation()

if __name__ == '__main__':
  main()