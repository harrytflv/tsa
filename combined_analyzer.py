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

  def grade_prediction(self, w_type):
    with open('cache/new_user_mapping.json', 'r') as f_in:
      user_map = json.load(f_in)
    stu_grades  = self.pr_analyzer.iteration_grades()
    stu2grad = {}
    for k, v in stu_grades.items():
      if k in user_map:
        stu2grad[user_map[k]] = v
    stu_commits = self.iteration_workload(w_type)
    stu2cmit = {}
    for k, v in stu_commits.items():
      if k in user_map:
        stu2cmit[user_map[k]] = v
    iter_points, iter_num = self.pt_analyzer.iteration_points(reload=True)
    stu2pnts = {}
    for k, v in iter_points.items():
      if k in user_map:
        stu2pnts[user_map[k]] = v
    return stu2grad, stu2cmit, stu2pnts

  def iteration_workload(self, w_type):
    iterations = self.gt_analyzer.iteration_commits()
    stu_map, proj_total = defaultdict(lambda: np.zeros((4,))), defaultdict(lambda: np.zeros((4,)))
    stu2proj = {}
    for i in range(4):
      for commit in iterations[i]:
        stu = commit['commit']['author']['name']
        proj = self.gt_analyzer.extract_proj(commit)
        info = self._get_info(commit, w_type)

        stu2proj[stu] = proj
        stu_map[stu][i] += info
        proj_total[proj][i] += info
    if 'normalized' in w_type:
      stu_map = {k: v/proj_total[stu2proj[k]] for k, v in stu_map.items()}
    return stu_map

  def prediction(self, w_type):
    from sklearn.linear_model import LinearRegression
    from sklearn.naive_bayes import GaussianNB
    stu2grad, stu2cmit, stu2pnts = self.grade_prediction(w_type)
    avg_grade = np.average([v for _, v in stu2grad.items()])
    set_stu = set(stu2grad.keys()) & set(stu2cmit.keys()) & set(stu2pnts.keys())
    print('Total number of students: {}'.format(len(set_stu)))

    linear_result = []
    for k, v in stu2grad.items():
      if k in set_stu:
        linear_model = LinearRegression()
        linear_model.fit(np.array([[1.0], [2.0], [3.0]]), v[:3])
        linear_result.append((linear_model.predict([[4]])[0]-v[3])**2)
    print(np.average(linear_result))
    fig, ax = plt.subplots()
    plotdata = pd.Series(linear_result, name='squared error')
    sns.distplot(plotdata)
    plt.savefig('results/linear_prediction.png')
    plt.close(fig)

    training_feature, training_label = [], []
    predict_feature, predict_label = [], []
    for k in set_stu:
      tmp_grad, tmp_cmit, tmp_pnts = stu2grad[k], stu2cmit[k], stu2pnts[k]
      for ite in range(3):
        training_feature.append([tmp_cmit[ite], tmp_pnts[ite]])
        # training_label.append(np.rint(tmp_grad[ite]))
        training_label.append(tmp_grad[ite])
      predict_feature.append([tmp_cmit[3], tmp_pnts[3]])
      predict_label.append(tmp_grad[3])
    # nb_model = GaussianNB()
    model = LinearRegression()
    model.fit(np.array(training_feature), np.array(training_label))
    predicts = model.predict(np.array(predict_feature))
    result = (predicts-np.array(predict_label))**2
    print(np.average(result))
    print(model.coef_)
    fig, ax = plt.subplots()
    plotdata = pd.Series(result, name='squared error')
    sns.distplot(plotdata)
    plt.savefig('results/linear_gt_pt_prediction.png')
    plt.close(fig)

    training_feature, training_label = [], []
    predict_feature, predict_label = [], []
    for k in set_stu:
      tmp_grad, tmp_cmit, tmp_pnts = stu2grad[k], stu2cmit[k], stu2pnts[k]
      tmp_avg = [avg_grade] + [np.average(tmp_grad[:i+1]) for i in range(3)]
      for ite in range(3):
        training_feature.append([tmp_cmit[ite], tmp_pnts[ite], tmp_avg[ite]])
        # training_label.append(np.rint(tmp_grad[ite]))
        training_label.append(tmp_grad[ite])
      predict_feature.append([tmp_cmit[3], tmp_pnts[3], tmp_avg[3]])
      predict_label.append(tmp_grad[3])
    # nb_model = GaussianNB()
    model = LinearRegression()
    model.fit(np.array(training_feature), np.array(training_label))
    predicts = model.predict(np.array(predict_feature))
    result = (predicts-np.array(predict_label))**2
    print(np.average(result))
    print(model.coef_)
    fig, ax = plt.subplots()
    plotdata = pd.Series(result, name='squared error')
    sns.distplot(plotdata)
    plt.savefig('results/linear_gt_pt_avg_prediction.png')
    plt.close(fig)

    mse1, mse2, mse3 = [], [], []
    for _, v in stu2grad.items():
      vv = np.array(v)
      mse1.append(np.sum((vv[1:]-np.average(vv[:1]))**2))
      mse2.append(np.sum((vv[2:]-np.average(vv[:2]))**2))
      mse3.append(np.sum((vv[3:]-np.average(vv[:3]))**2))
    print(np.average(mse1))
    print(np.average(mse2))
    print(np.average(mse3))


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
  # analyzer.workload_correlation_plot(w_type='file_edit_normalized')
  # analyzer.workload_correlation_plot(w_type='line_edit')
  analyzer.prediction('file_edit_normalized')

if __name__ == '__main__':
  main()