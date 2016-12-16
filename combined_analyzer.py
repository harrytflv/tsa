#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

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
        cmit.append(len(user_commits[user]))
        pnts.append(id2pnts[user])
        grad.append(id2grad[user])
        scor.append(id2scor[user])
        ids.append(user)
    print('Num Users: {}'.format(len(ids)))
    return cmit, pnts, grad, scor, ids

  def workload_correlation_plot(self):
    cmit, pnts, grad, scor, _ = self.workload_correlation()
    plotdata = pd.DataFrame({'commits':cmit,
                             'frac points':pnts,
                             'avg grade': grad,
                             'avg score': scor})
    fig, ax = plt.subplots()
    sns.pairplot(plotdata, kind='reg', vars=['commits', 'frac points', 'avg grade'])
    plt.savefig('results/combined_correlation.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.jointplot(x='commits', y='frac points', data=plotdata, kind='reg')
    plt.savefig('results/commit_pnts_correlation.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.jointplot(x='commits', y='avg grade', data=plotdata, kind='reg')
    plt.savefig('results/commit_grad_correlation.png')
    plt.close(fig)

def main():
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)

  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  peer_review = PeerReview('data/', 'peer_combined')

  gt_analyzer = GithubAnalyzer(tokens['github']['token'], project_info)
  pt_analyzer = PivotalTrackerAnalyzer(tokens['pivotal_tracker']['token'], project_info)
  pr_analyzer = PeerReviewAnalyzer(peer_review)

  analyzer = CombinedAnalyzer(gt_analyzer=gt_analyzer, pt_analyzer=pt_analyzer, pr_analyzer=pr_analyzer)
  analyzer.workload_correlation_plot()

if __name__ == '__main__':
  main()