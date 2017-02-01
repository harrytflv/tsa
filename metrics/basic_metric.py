#!/usr/bin/env 

class BasicMetric(object):
  """The base class for all metrics"""
  def __init__(self, proj, token):
    super(BasicMetric, self).__init__()
    self.proj = proj
    self.token = token
    self.ROOT_PATH = '/Users/Joe/Projects/TeamScope/analysis'
    self._load_connection()

  def _load_connection(self):
    raise NotImplementedError
    
def main():
  pass

if __name__ == '__main__':
  main()