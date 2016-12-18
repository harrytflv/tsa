#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from collections import defaultdict

from github_analyzer import GithubAnalyzer
from peer_review import PeerReviewAnalyzer
from pivotal_tracker_analyzer import PivotalTrackerAnalyzer
from data_util.data_reader import ProjectInfo, PeerReview

class CombinedAnalyzer(object):
  """docstring for CombinedAnalyzer"""
  def __init__(self, **arg):
    super(CombinedAnalyzer, self).__init__()
    self.gt_analyzer = arg.pop('gt_analyzer', None)
    self.pt_analyzer = arg.pop('pt_analyzer', None)
    self.pr_analyzer = arg.pop('pr_analyzer', None)

  def workload_correlation(self):
    from pt_pr_comparison import PtPrComparisonAnalyzer
    pt_pr_analyzer = PtPrComparisonAnalyzer(self.pr_analyzer, self.pt_analyzer)
    user_commits = self.gt_analyzer.user_commits()
    id2pnts, id2grad, id2scor = pt_pr_analyzer.consistency()
    cmit, pnts, grad, scor = [], [], [], []
    ids = []
    for user in user_commits:
      if user in id2pnts and user in id2grad and user in id2scor:
        if len(user_commits[user]) > 200:
          continue
        cmit.append(user_commits[user])
        pnts.append(id2pnts[user])
        grad.append(id2grad[user])
        scor.append(id2scor[user])
        ids.append(user)
    print('Num Users: {}'.format(len(ids)))
    return cmit, pnts, grad, scor, ids

  def workload_correlation_plot(self, w_type='num_commits'):
    self.proj_info = defaultdict(lambda: 0)
    cmit, pnts, grad, scor, _ = self.workload_correlation()
    self._get_project_total(cmit, w_type)
    plotdata = pd.DataFrame({w_type:[self._workload(item, w_type) for item in cmit],
                             'frac points':pnts,
                             'avg grade': grad,
                             'avg score': scor})
    fig, ax = plt.subplots()
    sns.pairplot(plotdata, kind='reg', vars=[w_type, 'frac points', 'avg grade'])
    plt.savefig('results/combined_correlation_{}.png'.format(w_type))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.jointplot(x=w_type, y='frac points', data=plotdata, kind='reg')
    plt.savefig('results/commit_pnts_correlation_{}.png'.format(w_type))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.jointplot(x=w_type, y='avg grade', data=plotdata, kind='reg')
    plt.savefig('results/commit_grad_correlation_{}.png'.format(w_type))
    plt.close(fig)

  def _workload(self, commits, w_type):
    info = np.sum([self._get_info(commit, w_type) for commit in commits])
    if 'normalized' in w_type:
      proj = self.gt_analyzer.extract_proj(commits[0])
      info /= self.proj_info[proj]
    return info

  def _get_project_total(self, cmit, w_type):
    for commits in cmit:
      for commit in commits:
        proj = self.gt_analyzer.extract_proj(commit)
        info = self._get_info(commit, w_type)
        self.proj_info[proj] += info

  def _get_info(self, commit, w_type):
    if 'num_commits' in w_type:
      return 1
    elif 'file_edit' in w_type:
      info = self.gt_analyzer.get_commit(commit['sha'])
      return len(info['files'])
    elif 'line_edit' in w_type:
      info = self.gt_analyzer.get_commit(commit['sha'])
      return np.sum([f['changes'] for f in info['files']])
    else:
      print('Unrecognizable w_type.')
      return 0

  def grade_prediction(self):
    for 
    self.pr_analyzer.

def main():
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)

  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  peer_review = PeerReview('data/', 'peer_combined')

  gt_analyzer = GithubAnalyzer(tokens['github']['token'], project_info)
  pt_analyzer = PivotalTrackerAnalyzer(tokens['pivotal_tracker']['token'], project_info)
  pr_analyzer = PeerReviewAnalyzer(peer_review)

  analyzer = CombinedAnalyzer(gt_analyzer=gt_analyzer, pt_analyzer=pt_analyzer, pr_analyzer=pr_analyzer)
  # analyzer.workload_correlation_plot(w_type='num_commits_normalized')
  analyzer.workload_correlation_plot(w_type='file_edit_normalized')
  # analyzer.workload_correlation_plot(w_type='line_edit')

if __name__ == '__main__':
  main()