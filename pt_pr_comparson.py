#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import nltk

from data_util.data_reader import ProjectInfo, PeerReview
from peer_review import PeerReviewAnalyzer
from pivotal_tracker_analyzer import PivotalTrackerAnalyzer
from pivotal_tracker_api.tracker_api import TrackerApi

class PtPrComparisonAnalyzer(object):
  def __init__(self, pr_analyzer, pt_analyzer):
    self.pr_analyzer = pr_analyzer
    self.pt_analyzer = pt_analyzer

  def generate_student_map(self):
    dictPnts, dictNum = self.pt_analyzer.story_assign(reload=True)
    self.usernames, missing_id = [], []
    for k in dictPnts:
      if not k[-1] == '#':
        self.usernames.append(k)
      else:
        missing_id.append((k[:-2], k[-2]))

    self.student_list = list(self.pr_analyzer.dataset.students())
    self.map_list, counter = {}, 0
    for s in self.usernames:
      ss = self._nearest_neighbor(s)
      c = input('{} and {}?'.format(s, ss))
      if c == 'Y':
        self.map_list[s] = counter
        self.map_list[ss] = counter
        counter += 1
    with open('cache/user_mapping.json', 'w') as f_out:
      json.dump(self.map_list, f_out)
    print(counter)

  def _nearest_neighbor(self, wd):
    idx = np.argmin([self._distance(item, wd) for item in self.student_list])
    return self.student_list[idx]

  def _distance(self, wd_1, wd_2):
    wd_2 = ', '.join(reversed(wd_2.split(' ')))
    return nltk.edit_distance(wd_1.lower(), wd_2.lower())

  def consistency(self):
    with open('cache/user_mapping.json', 'r') as f_in:
      self.user_map = json.load(f_in)
    dictPnts, dictNum = self.pt_analyzer.story_assign(reload=True)
    grades = self.pr_analyzer.avg_grade()
    scores = self.pr_analyzer.avg_score(filtered=False)

    dictId2Pnts, dictId2Grad, dictId2Scor = {}, {}, {}
    for k, v in dictPnts.items():
      if k in self.user_map:
        dictId2Pnts[self.user_map[k]] = v
    for k, v in grades.items():
      if k in self.user_map:
        dictId2Grad[self.user_map[k]] = v
    for k, v in scores.items():
      if k in self.user_map:
        dictId2Scor[self.user_map[k]] = v
    return dictId2Pnts, dictId2Grad, dictId2Scor

  def consistency_plot(self):
    dictId2Pnts, dictId2Grad, dictId2Scor = self.consistency()
    lstGrad, lstScor = [], []
    for k, v in dictId2Pnts.items():
      if k in dictId2Grad:
        lstGrad.append((v, dictId2Grad[k]))
      else:
        print('Missing key Grade: {}'.format(k))
      if k in dictId2Scor:
        lstScor.append((v, dictId2Scor[k]))
      else:
        print('Missing key Score: {}'.format(k))

    df_grad = pd.DataFrame({'points': [item[0] for item in lstGrad], 'grade': [item[1] for item in lstGrad]})
    df_scor = pd.DataFrame({'points': [item[0] for item in lstScor], 'score': [item[1] for item in lstScor]})

    fig, ax = plt.subplots()
    sns.jointplot(x='points', y='grade', data=df_grad, kind="reg")
    plt.savefig('results/correlation_grade.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.jointplot(x='points', y='score', data=df_scor, kind="reg")
    plt.savefig('results/correlation_score.png')
    plt.close(fig)

def main():
  pr_data = PeerReview('data/', 'peer_combined')
  pr_analyzer = PeerReviewAnalyzer(pr_data)
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)  
  proj_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  pt_analyzer = PivotalTrackerAnalyzer(proj_info, tokens['pivotal_tracker']['token'])
  analyzer = PtPrComparisonAnalyzer(pr_analyzer, pt_analyzer)
  # analyzer.generate_student_map()
  analyzer.consistency_plot()

if __name__ == '__main__':
  main()