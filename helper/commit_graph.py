#!/usr/bin/env 

import numpy as np
import json
import datetime

class CommitGraph(object):
  """the Commit Graph"""
  def __init__(self):
    super(CommitGraph, self).__init__()
  
  def construct(self, sha2cmit):
    """
      Build the commit graph for a project

      Input
        - commits: all commits of interest from the project
    """
    self.root = []
    self.sha2node = {k:CommitGraphNode(v) for k, v in sha2cmit.items()}
    for k in sha2cmit:
      parents = sha2cmit[k]['parents']
      tnode = self.sha2node[k]
      for p in parents:
        if p['sha'] in self.sha2node:
          pnode = self.sha2node[p['sha']]
          tnode.parents.append(pnode)
          pnode.children.append(tnode)
      if len(parents) == 0:
        self.root.append(self.sha2node[k])

class CommitGraphNode(object):
  """Node in the commit graph"""
  def __init__(self, commit):
    super(CommitGraphNode, self).__init__()
    self.commit = commit
    self.parents = []
    self.children = []
    self.type = self._get_type(commit)
    self.timestamp = datetime.datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ')
    self._nsignature = None
    self._next = None

  def next(self, step, get_type):
    """
      Get a subgraph starting from this node and going further k step.
      A child node is only considered when it has different type from its parent.
      All types will be considered. All branches will be considered.

      Input
        - step: how many steps going forward
        - get_type: a function transform initial type into used types
      Output
        - a list of all linear subgraphs
    """
    if self._next:
      return self._next
    threads = [([get_type(f_type)], self) for f_type in self.type]
    end_list = []
    for thread in threads:
      nd, ctype = thread[1], thread[0][-1]
      if len(thread[0]) > step or len(nd.children)==0:
        end_list.append(thread[0])
        continue

      for cnode in nd.children:
        for ntype in set([get_type(t) for t in cnode.type]):
          if ntype == ctype:
            threads.append((thread[0], cnode))
          else:
            threads.append((thread[0]+[ntype], cnode))
    self._next = end_list
    return end_list

  def next_signature(self, step, get_type):
    """
      Get a subgraph signature of the node. A signature means it's unique for the subgraph information.
      Subgraph information will be got from next function.

      Input
        - step, get_type: parameters that will be passed to self.next()
      Output
        - a string signature
    """
    if not self._nsignature:
      self._nsignature = '|'.join(sorted(['-'.join(ptn) for ptn in self.next(step, get_type)]))
    return self._nsignature

  def _get_type(self, commit):
    """
      Get type of a commit

      Input
        - commit: the commit object returned from GitHub get single commit API
      Output
        - a list of types, converted from files
    """
    if len(commit['parents']) > 1:
      return ['merge']
    types = [self._file_type(f) for f in commit['files']]
    return list(filter(lambda x: x, types))

  def _file_type(self, ifile):
    """
      Infer the type of file based on the input

      Input
        - ifile: a dictionary returned from GitHub API
    """
    finfo = ifile['filename'].split('/')
    if finfo[0] == 'app':
      return 'app {}'.format(finfo[1])
    if finfo[0] == 'features':
      return 'test behavior'
    if finfo[0] == 'spec':
      return 'test unit'
    if finfo[0] == 'test':
      return 'test'
    if finfo[0] == 'db':
      return 'database'
    if finfo[0].lower() in ['gemfile', 'readme.md', '.gitignore', 'config']:
      return 'config'
    return None

def main():
  with open('../cache/sha2commit_new.json', 'r') as f_in:
    commit_cache = json.load(f_in)
  graph = CommitGraph()
  graph.construct(commit_cache)
  print(len(graph.root))
  counter = 0
  for k, n in graph.sha2node.items():
    if len(n.parents) == 0:
      counter += 1
  print(counter)

if __name__ == '__main__':
  main()