#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import json
import os
from collections import defaultdict
from tqdm import tqdm

from github_analyzer import GithubAnalyzer
from github_api.github_api import GithubApi
from data_util.data_reader import ProjectInfo
from helper.commit_graph import CommitGraph

class ProcessMiningAnalyzer(object):
  """
    Process mining based on git commits.
  """
  def __init__(self, project_info):
    """
      Tokens are read from conf/tokens.json file.
      
      Input
        - project_info: dataset contains all course projects
    """
    super(ProcessMiningAnalyzer, self).__init__()
    self.project_info = project_info
    with open('conf/tokens.json', 'r') as f_in:
      self.token = json.load(f_in)
    self.gt_analyzer = GithubAnalyzer(self.token['github']['token'], self.project_info)
    self.gt_client = GithubApi(self.token['github']['token'])
    self.out_header = 'process_mining'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)

  def ftype_count(self):
    self._build_graph()
    types, mixed_numbers = [], []
    for _, v in self.commit_graph.sha2node.items():
      valid_types = set()
      for f_type in v.type:
        if 'app' in f_type:
          valid_type = 'app'
        elif 'test' in f_type:
          valid_type = 'test'
        else:
          valid_type = f_type
        valid_types.add(valid_type
          )
        types.append(valid_type)
      if 'unknown' in valid_types:
        valid_types.remove('unknown')
      mixed_numbers.append(len(valid_types))

    plotdata = pd.DataFrame({'types': types})
    fig, ax = plt.subplots()
    sns.countplot(y='types', data=plotdata)
    plt.savefig('{}/type_count.png'.format(self.out_header))
    plt.close(fig)

    plotdata = pd.DataFrame({'mixed_numbers': mixed_numbers})
    fig, ax = plt.subplots()
    sns.countplot(y='mixed_numbers', data=plotdata)
    plt.savefig('{}/mixed_number_count.png'.format(self.out_header))
    plt.close(fig)

  def frequent_pattern(self, step_size=3):
    import time
    def get_type(ftype):
      return ftype[0].upper()
    self._build_graph()
    patterns, counter = [], defaultdict(lambda: 0)
    time_atm, time_tam, time_all = [], [], []
    for _, nd in tqdm(self.commit_graph.sha2node.items()):
      for ptn in nd.next(step_size, get_type):
        if len(ptn) != step_size+1:
          continue
        str_id = '-'.join(ptn)
        patterns.append(str_id)
        counter[str_id] += 1
        if 'A-T' in str_id:
          time_atm.append(time.mktime(nd.timestamp.timetuple()))
        if 'T-A' in str_id:
          time_tam.append(time.mktime(nd.timestamp.timetuple()))
        time_all.append(time.mktime(nd.timestamp.timetuple()))

    sorted_list = sorted([(k, v) for k, v in counter.items()], key=lambda x: -x[1])
    print(sorted_list[:10])

    plotdata = pd.DataFrame({'patterns':patterns})
    fig, ax = plt.subplots()
    sns.countplot(y='patterns', data=plotdata)
    plt.savefig('{}/pattern_count.png'.format(self.out_header))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.distplot([np.log(v) for _, v in counter.items()])
    plt.savefig('{}/pattern_count_dist.png'.format(self.out_header))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.distplot(time_atm, label='at')
    sns.distplot(time_tam, label='ta')
    plt.legend()
    plt.savefig('{}/hist_tam_atm.png'.format(self.out_header))
    plt.close(fig)

    ite_at, ite_ta = [], []
    ite_boundary = [1.42e9,1.45e9,1.47e9, 1.50e9]
    def boundary_func(x):
      return x > ite_boundary[i] and x < ite_boundary[i+1]
    for i in range(3):
      num_ite = len(list(filter(boundary_func, time_all)))
      num_at = len(list(filter(boundary_func, time_atm)))
      num_ta = len(list(filter(boundary_func, time_tam)))

      ite_at.append(num_at / num_ite)
      ite_ta.append(num_ta / num_ite)
    print(ite_at)
    print(ite_ta)

  def neighbor_selection(self, step_size=3):
    import time
    import time
    def get_type(ftype):
      return ftype[0].upper()
    self._build_graph()
    patterns, edge_counter = defaultdict(lambda: 0), defaultdict(lambda: 0)
    total_num = len(self.commit_graph.sha2node)
    for _, nd in tqdm(self.commit_graph.sha2node.items()):
      tmp_signature = nd.next(step_size, get_type)
      for ptn in tmp_signature:
        str_id = '-'.join(ptn)
        patterns[str_id] += 1
        if nd.parents:
          for pnd in nd.parents:
            for pptn in pnd.next(step_size, get_type):
              edge_counter['{}->{}'.format('-'.join(pptn), str_id)] += 1
        else:
          patterns['START'] += 1
          edge_counter['START->{}'.format(str_id)] += 1

    print(len(patterns))
    with open('cache/pattern_count.json', 'w') as f_out:
      json.dump(patterns, f_out)
    with open('cache/edge_strength.json', 'w') as f_out:
      json.dump(edge_counter, f_out)

    filtered_patterns = {}
    for k, v in patterns.items():
      if np.log(v) > -1:
        filtered_patterns[k] = v
    # print(sorted(filtered_patterns, key=lambda x: -filtered_patterns[x]))
    print(len(filtered_patterns))

    fig, ax = plt.subplots()
    sns.distplot([np.log(v) for _, v in patterns.items()])
    plt.savefig('{}/state_counter.png'.format(self.out_header))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.distplot([v for _, v in filtered_patterns.items()])
    plt.savefig('{}/freq_state_counter.png'.format(self.out_header))
    plt.close(fig)

    ptn2index = {}
    for k in filtered_patterns:
      ptn2index[k] = len(ptn2index)
    edge_mat = np.zeros((len(ptn2index), len(ptn2index)))
    for k, v in edge_counter.items():
      ptn_1, ptn_2 = k.split('->')
      if ptn_1 in ptn2index and ptn_2 in ptn2index:
        edge_mat[ptn2index[ptn_1], ptn2index[ptn_2]] = v

    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0)
    y = model.fit_transform(edge_mat)

    fig, ax = plt.subplots()
    plt.scatter([x[0] for x in y], [x[1] for x in y])
    plt.savefig('{}/filtered_link_mat_tsne.png'.format(self.out_header))
    plt.close(fig)


  def _build_graph(self, project=None):
    """
      Build a commit graph for a given project. If project is None, build a single graph for all projects.

      Input
        - project: a project from project info 
    """
    self.commit_graph = CommitGraph()
    with open('cache/sha2commit_new.json', 'r') as f_in:
      sha2cmit = json.load(f_in)
    proj_dict = {}
    if project:
      commits = self.gt_client.get_commits(project['repo']['owner'], project['repo']['repo'])
      for cmit in commits:
        sha = cmit['sha']
        if sha in sha2cmit:
          proj_dict[sha] = sha2cmit[sha]
    else:
      proj_dict = sha2cmit
    self.commit_graph.construct(proj_dict)
    if project:
      print('{}: {} root'.format(project['project'], len(self.commit_graph.root)))
    else:
      print('All: {} root'.format(len(self.commit_graph.root)))

  def _convert_commit(self, commit):
    """
      Convert a commit into an object for analysis

      Input
        - commit: a dictionary got from GitHub get single commit API
    """
    file_types = [self._file_type(item) for item in commit['files']]

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  analyzer = ProcessMiningAnalyzer(project_info)
  # analyzer.ftype_count()
  # analyzer.frequent_pattern()
  analyzer.neighbor_selection()

if __name__ == '__main__':
  main()