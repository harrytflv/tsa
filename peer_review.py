#!/usr/bin/env 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from collections import defaultdict

from data_util.data_reader import PeerReview

class PeerReviewAnalyzer(object):
  '''Used for analyzing peer reviews'''
  def __init__(self, dataset):
    super(PeerReviewAnalyzer, self).__init__()
    self.dataset = dataset

  def token_freq(self):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    token_counter = defaultdict(lambda: 0)
    stopwords = nltk.corpus.stopwords.words('english')
    for item in self.dataset:
      for i in [1, 2, 3, 4, 5, 6]:
        key = 'Comments about Person {}:'.format(i)
        for token in tokenizer.tokenize(item[key]):
          if not token in stopwords:
            token_counter[token] += 1
    token_counter = [(k, v) for k, v in token_counter.items()]
    token_counter = sorted(token_counter, key=lambda x: -x[1])
    plotdata = pd.DataFrame.from_records(token_counter[:100], columns=['token', 'count'])
    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(10, 30))
    sns.set_style("whitegrid")
    sns.barplot(y="token", x="count", data=plotdata)
    plt.savefig('results/token_freq.png')
    plt.close(fig)

  def sentiment_analysis(self, dataset, filtered):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk import tokenize

    sid = SentimentIntensityAnalyzer()
    sentences, grading, scores, students = [], [], [], []

    for item in dataset:
      for i in [1, 2, 3, 4, 5, 6]:
        key = 'Comments about Person {}:'.format(i)
        grade = item['Rating for Person {}:'.format(i)]
        stu_key = 'What is your name? (Person 1)'.format(i) if i == 1 else 'Person {}:'.format(i)
        student = item[stu_key]
        if grade:
          for sent in tokenize.sent_tokenize(item[key]):
            score = sid.polarity_scores(sent)['compound']
            if (filtered and np.abs(score) > .01) or not filtered:
              sentences.append(sent)
              grading.append(int(grade))
              scores.append(score)
              students.append(student)

    print(len(sentences))
    return sentences, grading, scores, students

  def sentiment_analysis_single(self, filtered=False):
    sentences, grading, scores, _ = self.sentiment_analysis(self.dataset, filtered)

    fig, ax = plt.subplots()
    plotdata = pd.DataFrame({'grade':grading, 'score':scores})
    ax = sns.jointplot(x="grade", y="score", data=plotdata, kind="reg", x_estimator=np.mean)
    f_name = 'results/sent.png' if not filtered else 'results/sent_filtered.png'
    plt.savefig(f_name)
    plt.close(fig)

  def sentiment_analysis_iteration(self, filtered=False):
    frames = []
    for ite in range(4):
      sentences, grading, scores, _ = self.sentiment_analysis(self.dataset.iterations[ite], filtered)
      frames.append(pd.DataFrame({'grade':grading, 'score':scores, 'iteration':ite+1}))
    plotdata = pd.concat(frames)

    fig, ax = plt.subplots()
    sns.lmplot(x='grade', y='score', hue='iteration', data=plotdata, x_estimator=np.mean)
    f_name = 'results/sent_iterations.png' if not filtered else 'results/sent_iterations_filtered.png'
    plt.savefig(f_name)
    plt.close(fig)

  def consistency(self, filtered=False, divide_by_iterations=True):
    if divide_by_iterations:
      grading, scores, students = [], [], []
      for ite in range(4):
        _, tmp_g, tmp_s, tmp_st = self.sentiment_analysis(self.dataset.iterations[ite], filtered)
        grading.extend(tmp_g)
        scores.extend(tmp_s)
        students.extend([stu+str(ite) for stu in tmp_st])
    else:
      _, grading, scores, students = self.sentiment_analysis(self.dataset, filtered)
    libStu2Grade, libStu2Score = defaultdict(lambda: []), defaultdict(lambda: [])
    for ite, stu in enumerate(students):
      libStu2Grade[stu].append((grading[ite]-1.0)/4.0)
      libStu2Score[stu].append((scores[ite]+1.0)/2.0)
    lstStdGrade, lstAvgGrade = [], []
    lstStdScore, lstAvgScore = [], []
    for s in set(students):
      lstStdGrade.append(np.std(libStu2Grade[s]))
      lstAvgGrade.append(np.average(libStu2Grade[s]))
      lstStdScore.append(np.std(libStu2Score[s]))
      lstAvgScore.append(np.average(libStu2Score[s]))

    fig, ax = plt.subplots()
    sns.distplot(lstStdGrade, label='grade')
    sns.distplot(lstStdScore, label='score')
    plt.legend()
    f_name = 'results/consistency.png' if not filtered else 'results/consistency_filtered.png'
    plt.savefig(f_name)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.distplot(lstAvgGrade, label='grade')
    sns.distplot(lstAvgScore, label='score')
    plt.legend()
    f_name = 'results/grading.png' if not filtered else 'results/grading_filtered.png'
    plt.savefig(f_name)
    plt.close(fig)

  def consistency_more_grade(self, filtered=False, divide_by_iterations=True, plot=True):
    if divide_by_iterations:
      grading, scores, students = [], [], []
      for ite in range(4):
        _, tmp_g, tmp_s, tmp_st = self.sentiment_analysis(self.dataset.iterations[ite], filtered)
        grading.extend(tmp_g)
        scores.extend(tmp_s)
        students.extend([stu+str(ite) for stu in tmp_st])
    else:
      _, grading, scores, students = self.sentiment_analysis(self.dataset, filtered)

    libStu2Grade, libStu2Score = defaultdict(lambda: []), defaultdict(lambda: [])
    for ite, stu in enumerate(students):
      libStu2Score[stu].append((scores[ite]+1.0)/2.0)

    for ite, dataset in enumerate(self.dataset.iterations):
      for item in dataset:
        for i in [1,2,3,4,5,6]:
          grade = item['Rating for Person {}:'.format(i)]
          stu_key = 'What is your name? (Person 1)'.format(i) if i == 1 else 'Person {}:'.format(i)
          if grade:
            libStu2Grade[item[stu_key]+str(ite)].append((int(grade)-1.0)/4.0)

    lstStdGrade, lstAvgGrade = [], []
    lstStdScore, lstAvgScore = [], []
    for s in set(students):
      lstStdGrade.append(np.std(libStu2Grade[s]))
      lstAvgGrade.append(np.average(libStu2Grade[s]))
      lstStdScore.append(np.std(libStu2Score[s]))
      lstAvgScore.append(np.average(libStu2Score[s]))

    fig, ax = plt.subplots()
    sns.distplot(lstStdGrade, label='grade')
    sns.distplot(lstStdScore, label='score')
    plt.legend()
    f_name = 'results/consistency_unfair.png' if not filtered else 'results/consistency_unfair_filtered.png'
    plt.savefig(f_name)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.distplot(lstAvgGrade, label='grade')
    sns.distplot(lstAvgScore, label='score')
    plt.legend()
    f_name = 'results/grading_unfair.png' if not filtered else 'results/grading_unfair_filtered.png'
    plt.savefig(f_name)
    plt.close(fig)

  def avg_grade(self):
    libStu2Grade= defaultdict(lambda: [])
    for item in self.dataset:
      for i in [1,2,3,4,5,6]:
        grade = item['Rating for Person {}:'.format(i)]
        stu_key = 'What is your name? (Person 1)'.format(i) if i == 1 else 'Person {}:'.format(i)
        if grade:
          libStu2Grade[item[stu_key]].append(int(grade))
    return {k: np.average(v) for k, v in libStu2Grade.items()}

  def avg_score(self, filtered=False):
    _, grading, scores, students = self.sentiment_analysis(self.dataset, filtered)
    libStu2Score = defaultdict(lambda: [])
    for ite, stu in enumerate(students):
      libStu2Score[stu].append(scores[ite])
    return {k: np.average(v) for k, v in libStu2Score.items()}

def main():
  # dataset = PeerReview('data/Peer Evaluation (Responses) - Iter1.csv', 'peer_single')
  dataset = PeerReview('data/', 'peer_combined')
  analyzer = PeerReviewAnalyzer(dataset)
  analyzer.consistency_more_grade()
  analyzer.consistency()
  # analyzer.sentiment_analysis_single()
  # analyzer.sentiment_analysis_iteration()
  # analyzer.token_freq()

if __name__ == '__main__':
  main()