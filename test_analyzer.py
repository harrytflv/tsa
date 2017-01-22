#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
import os
from tqdm import tqdm

from pivotal_tracker_api.tracker_api import TrackerApi
from data_util.data_reader import ProjectInfo

class TestAnalyzer(object):
  """
    Analysis based on test information extracted from TravisCI build history.
    Assume certain cache files exist.
  """
  def __init__(self, tokens, proj_info):
    super(TestAnalyzer, self).__init__()
    self.tokens = tokens
    self.project_info = proj_info
    self.pt_client = TrackerApi(tokens['pivotal_tracker']['token'])
    with open('cache/log_info.json', 'r') as f_in:
      self.build_info = json.load(f_in)
    self.out_header = 'test_analysis'
    if not self.out_header in os.listdir('results'):
      os.mkdir('results/{}'.format(self.out_header))
    self.out_header = 'results/{}'.format(self.out_header)

  def cucumber_scenarios(self, proj):
    """
      Cucumber scenario analysis for a given project.
      It correlates commits/builds with user stories.

      Input
        - proj: one project from project info dataset.
    """
    bd_history = self.build_info[proj['ID']]
    features, scenarios = [], []
    total_features, total_scenarios = set(), set()
    for bd in bd_history:
      tmp_feature, tmp_scenarios = set(), set()
      for scenario in bd['cucumber']['scenarios']:
        feature = self._extract_feature(scenario['feature'])
        scenario = self._extract_scenario(scenario['scenario'])
        if feature and scenario:
          tmp_feature.add(feature)
          tmp_scenarios.add(scenario)
          total_features.add(feature)
          total_scenarios.add(scenario)
      features.append(tmp_feature)
      scenarios.append(tmp_scenarios)

    stories = self.pt_client.get_stories(proj['tracker'])
    story_str = [self._get_story_str(story) for story in stories]
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    total_feature = list(total_features)
    word_corps = total_feature + story_str
    flags = ['cucumber']*len(total_features) + ['stories']*len(story_str)

    plotdata = pd.DataFrame({'length':[len(txt) for txt in word_corps], 'flag':flags})
    fig, ax = plt.subplots()
    sns.boxplot(x='flag', y='length', data=plotdata)
    plt.savefig('{}/{}_boxplot.png'.format(self.out_header, proj['ID']))
    plt.close(fig)

    word_vecs = vectorizer.fit_transform(word_corps)

    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(word_vecs)

    cucumber_vec_txt = list(zip(tfidf[:len(total_features)], total_feature))
    story_vec_txt = list(zip(tfidf[len(total_features):], story_str))
    best_match = []
    for vec_txt in cucumber_vec_txt:
      story = self._nearest_neighbor(vec_txt, story_vec_txt)
      best_match.append((vec_txt[1], story[1]))

    with open('{}/match_results.txt'.format(self.out_header), 'a') as f_out:
      for cucumber_txt, story_txt in best_match:
        f_out.write(cucumber_txt)
        f_out.write('\n---------------\n')
        f_out.write(story_txt)
        f_out.write('\n\n\n')

    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0)
    plot_points = model.fit_transform(tfidf.toarray())

    fig, ax = plt.subplots()
    plt.scatter([x[0] for x in plot_points], [x[1] for x in plot_points], c=['g' if x == 'cucumber' else 'b' for x in flags], alpha=0.5)
    plt.savefig('{}/{}_scatter.png'.format(self.out_header, proj['ID']))
    plt.close(fig)

  def _nearest_neighbor(self, pnt, candidate_list):
    choice = list(sorted(candidate_list, key=lambda x: self._distance(x, pnt)))[0]
    return choice

  def _distance(self, pnt_1, pnt_2):
    return np.linalg.norm(pnt_1[0].toarray()-pnt_2[0].toarray())

  def _get_story_str(self, story):
    """
      Extract a string description out of a user story
    """
    str_output = story['name']
    if 'description' in story:
      str_output += '\n'
      str_output += story['description']
    return str_output

  def _extract_feature(self, log_str):
    """
      Extract the feature from a log line
      All cucumber feature lines are in the format of
      "Feature: Create an admin account"
    """
    return log_str
    # if not log_str:
    #   return False
    # return log_str[9:]

  def _extract_scenario(self, log_str):
    """
      Extract the scenario from a log line
      All cucumber scenario lines are in the format of
      "  Scenario: An admin cancels editing their information # features/admins/edit.feature:15"
    """
    log_str = log_str.replace('\t', '')
    tmp_lst = log_str.split('#')
    index = 0 if 'Scenario' in tmp_lst[0] else 1
    tmp_lst = tmp_lst[index].split(' ')
    if not 'Scenario:' in tmp_lst:
      return False
    return ' '.join(tmp_lst[tmp_lst.index('Scenario:')+1:])

def main():
  project_info = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'proj-info')
  with open('conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  analyzer = TestAnalyzer(tokens, project_info)
  for proj in tqdm(project_info):
    analyzer.cucumber_scenarios(proj)

if __name__ == '__main__':
  main()