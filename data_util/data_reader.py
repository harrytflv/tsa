#!/usr/bin/env 

import numpy as np
import csv

class Dataset(list):
  def __init__(self, data_file, data_name):
    self.data_file = data_file
    self.data_name = data_name
    self._preload()

  def _preload(self):
    raise NotImplementedError()

  def print_metadata(self):
    print('Length: {}'.format(len(self)))

  def print_examples(self, num=5):
    for item in np.random.choice(self, num):
      self._print_single_item(item)

  def _print_single_item(self, item):
    print(item)

class PeerReview(Dataset):
  def _preload(self):
    if self.data_name == 'peer_single':
      with open(self.data_file, 'r') as f_in:
        csv_reader = csv.DictReader(f_in)
        for row in csv_reader:
          self.append(row)
    elif self.data_name == 'peer_combined':
      self.iterations = [[], [], [], []]
      for ite in [1, 2, 3, 4]:
        with open('{}Peer Evaluation (Responses) - Iter{}.csv'.format(self.data_file, ite), 'r') as f_in:
          csv_reader = csv.DictReader(f_in)
          for row in csv_reader:
            self.append(row)
            self.iterations[ite-1].append(row)
    else:
      print('Input data name not valid.')

  def students(self):
    set_students = set()
    for item in self:
      for i in [1, 2, 3, 4, 5, 6]:
        key = 'What is your name? (Person 1)'.format(i) if i == 1 else 'Person {}:'.format(i)
        set_students.add(item[key])
    return set_students

  def get_comments(self, item, index=-1):
    if index in [1, 2, 3, 4, 5, 6]:
      return item['Comments about Person {}:'.format(index)]
    return [item['Comments about Person {}:'.format(i)] for i in [1, 2, 3, 4, 5, 6]]

  def get_grades(self, item, index=-1):
    if index in [1, 2, 3, 4, 5, 6]:
      return item['Rating for Person {}:'.format(index)]
    return [item['Rating for Person {}:'.format(i)] for i in [1, 2, 3, 4, 5, 6]]

  def get_students(self, item, index=-1):
    if index in [1, 2, 3, 4, 5, 6]:
      stu_key = 'What is your name? (Person 1)'.format(index) if index == 1 else 'Person {}:'.format(index)
      return item[stu_key]
    students = []
    for item in [1, 2, 3, 4, 5, 6]:
      stu_key = 'What is your name? (Person 1)'.format(i) if i == 1 else 'Person {}:'.format(i)
      students.append(item[stu_key])
    return students

class ProjectInfo(Dataset):
  def _preload(self):
    with open(self.data_file, 'r') as f_in:
      csv_reader = csv.DictReader(f_in)
      for row in csv_reader:
        self.append({
          'semester': row['w'],
          'ID': row['Team#'],
          'project': row['Project'],
          'deployment': row['Deployment'],
          'repo': self._get_repo_info(row['Repo']),
          'code_climate': row['CodeClimate'],
          'tracker': row['Tracker'].split('/')[-1],
          'students': [item.strip() for item in row['Students'].split(',')]
        })

  def _get_repo_info(self, repo_url):
    info_lst = repo_url.split('/')
    index = info_lst.index('github.com')
    return {
      'owner': info_lst[index+1],
      'repo': info_lst[index+2]
    }

class IterationGrading(Dataset):
  def _preload(self):
    if self.data_name == 'cumulative':
      with open(self.data_file, 'r') as f_in:
        csv_reader = csv.DictReader(f_in)
        for row in csv_reader:
          self.append(row)
    else:
      for ite in range(4):
        for phase in range(2):
          f_name = '{}{}'.format(self.data_file, self.iter_filename((ite+1, phase+1)))
          with open(f_name, 'r') as f_in:
            csv_reader = csv.DictReader(f_in)
            for row in csv_reader:
              row['iteration'] = ite + 1
              row['phase'] = phase + 1
              self.append(row)

  def iter_filename(self, ite):
    return {
      (1, 1): 'Grading Sheet - Iter1-1 (CURR).csv',
      (1, 2): 'Grading Sheet - Iter1-2.csv',
      (2, 1): 'Grading Sheet - Iter2-1.csv',
      (2, 2): 'Grading Sheet - Iter2-2.csv',
      (3, 1): 'Grading Sheet - Iter3-1.csv',
      (3, 2): 'Grading Sheet - Iter3-2.csv',
      (4, 1): 'Grading Sheet - Iter4-1.csv',
      (4, 2): 'Grading Sheet - Iter4-2.csv',
    }[ite]

def main():
  dataset = PeerReview('data/Peer Evaluation (Responses) - Iter1.csv', 'peer_single')
  dataset.print_metadata()
  dataset.print_examples()

  dataset = PeerReview('data/', 'peer_combined')
  dataset.print_metadata()
  dataset.print_examples()

  dataset = ProjectInfo('data/CS 169 F16 Projects - Sheet1.csv', 'project-info')
  dataset.print_metadata()
  dataset.print_examples()

  dataset = IterationGrading('data/Grading Sheet - Cumulative Grading.csv', 'cumulative')
  dataset.print_metadata()
  dataset.print_examples()

  dataset = IterationGrading('data/', 'detailed')
  dataset.print_metadata()
  dataset.print_examples()


if __name__ == '__main__':
  main()
