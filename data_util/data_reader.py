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

class ProjectInfo(Dataset):
  def _preload(self):
    with open(self.data_file, 'r') as f_in:
      csv_reader = csv.DictReader(f_in)
      for row in csv_reader:
        self.append(row)

def main():
  dataset = PeerReview('../data/Peer Evaluation (Responses) - Iter1.csv', 'peer_single')
  dataset.print_metadata()
  dataset.print_examples()

  dataset = PeerReview('../data/', 'peer_combined')
  dataset.print_metadata()
  dataset.print_examples()

  dataset = ProjectInfo('../data/CS 169 F16 Projects - Sheet1.csv', 'project-info')
  dataset.print_metadata()
  dataset.print_examples()

if __name__ == '__main__':
  main()
