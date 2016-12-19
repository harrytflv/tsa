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

    # print(len(sentences))
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

  def tf_idf(self):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    from nltk import tokenize
    import string
    stemmer = PorterStemmer()
    translator = str.maketrans({key: None for key in string.punctuation})
    def tokenize_sentence(sent):
      sanitized = sent.lower().translate(translator)
      tokens = filter(lambda x: not x in stopwords.words('english'), tokenize.word_tokenize(sanitized))
      return [stemmer.stem(wd) for wd in tokens]
    sentences = []
    for item in self.dataset:
      for comment in self.dataset.get_comments(item):
        for sent in tokenize.sent_tokenize(comment):
          sentences.append(sent)
    tfidf = TfidfVectorizer(tokenizer=tokenize_sentence, stop_words='english')
    vecs = tfidf.fit_transform(sentences)
    return vecs, tfidf, sentences

  def tf_idf_plot(self):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    vecs, _, sentences= self.tf_idf()
    with open('results/sentences.csv', 'w') as f_out:
      f_out.write('\n'.join(sentences))

    cluster_model = KMeans(n_clusters=8)
    labels = cluster_model.fit_predict(vecs)
    for i in range(8):
      with open('results/clusters{}.txt'.format(i), 'w') as f_out:
        for sent, label in zip(sentences, labels):
          if label == i:
            f_out.write(sent)
            f_out.write('\n')

    model = TSNE(n_components=2)
    tsne_vec = model.fit_transform(vecs.toarray())
    plotdata = pd.DataFrame({'d1': tsne_vec[:,0], 'd2': tsne_vec[:,1], 'color': labels})

    fig, ax = plt.subplots()
    sns.lmplot('d1', 'd2', data=plotdata, hue='color', fit_reg=False)
    plt.savefig('results/tf_idf.png')
    plt.close(fig)

  def iteration_grades_scores(self, filtered=False):
    grading, scores, students = [], [], []
    for ite in range(4):
      _, tmp_g, tmp_s, tmp_st = self.sentiment_analysis(self.dataset.iterations[ite], filtered)
      grading.append(tmp_g)
      scores.append(tmp_s)
      students.append(tmp_st)
    return grading, scores, students

  def iteration_grades(self):
    stu2grad = defaultdict(lambda: [[], [], [], []])
    for ite, dataset in enumerate(self.dataset.iterations):
      for item in dataset:
        for i in [1,2,3,4,5,6]:
          grade = item['Rating for Person {}:'.format(i)]
          stu_key = 'What is your name? (Person 1)'.format(i) if i == 1 else 'Person {}:'.format(i)
          if grade:
            stu2grad[item[stu_key]][ite].append(int(grade))
    result = {}
    for k, v in stu2grad.items():
      if not [] in v:
        result[k] = [np.average(item) for item in v]
    return result
    # return {k: [np.average(ite) if len(ite)>0 else 0.0 for ite in v] for k, v in stu2grad.items()}

def main():
  # dataset = PeerReview('data/Peer Evaluation (Responses) - Iter1.csv', 'peer_single')
  dataset = PeerReview('data/', 'peer_combined')
  analyzer = PeerReviewAnalyzer(dataset)
  # analyzer.consistency_more_grade()
  # analyzer.consistency()
  # analyzer.sentiment_analysis_single()
  # analyzer.sentiment_analysis_iteration()
  # analyzer.token_freq()
  # analyzer.tf_idf_plot()

if __name__ == '__main__':
  main()