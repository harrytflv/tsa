#!/usr/bin/env 

import MySQLdb
import json

class MysqlApi(object):
  """API wrapper for mysql database"""
  def __init__(self, config):
    super(MysqlApi, self).__init__()
    self.config = config

  def get_connection(self):
    self.db = MySQLdb.connect(host=self.config['host'],
                               user=self.config['user'],
                               passwd=self.config['passwd'],
                               db=self.config['db'])
    self.cur = self.db.cursor()

  def disconnect(self):
    self.db.close()

  def get_projects(self):
    """
      Get all projects in the database.
    """
    self.get_connection()
    self.cur.execute('SELECT * FROM projects;')
    projects = [Projects(row) for row in self.cur.fetchall()]
    self.disconnect()
    return projects

  def get_watchers(self, proj):
    """
      Get all watchers of a project.
    """
    self.get_connection()
    self.cur.execute('SELECT * FROM watchers WHERE repo_id={}'.format(proj.id))
    watchers = [Watchers(row) for row in self.cur.fetchall()]
    self.disconnect()
    return watchers

  def get_issues(self, proj):
    """
      Get all issues of a project
    """
    self.get_connection()
    self.cur.execute('SELECT * FROM issues WHERE repo_id={}'.format(proj.id))
    issues = [Issues(row) for row in self.cur.fetchall()]
    self.disconnect()
    return issues

class MysqlTables(object):
  """Class templates for mysql tables"""
  def __init__(self):
    super(MysqlTables, self).__init__()    

class Projects(MysqlTables):
  def __init__(self, row):
    super(Projects, self).__init__()
    self.id = row[0]
    self.url = row[1]
    self.owner_id = row[2]
    self.name = row[3]
    self.description = row[4]
    self.language = row[5]
    self.created_at = row[6]
    self.ext_ref_id = row[7]
    self.forked_from = row[8]
    self.deleted = row[9]

class Watchers(MysqlTables):
  def __init__(self, row):
    super(Watchers, self).__init__()
    self.repo_id = row[0]
    self.user_id = row[1]
    self.created_at = row[2]
    self.ext_ref_id = row[3]

class Issues(MysqlTables):
  def __init__(self, row):
    super(Issues, self).__init__()
    self.id = row[0]
    self.repo_id = row[1]
    self.reporter_id = row[2]
    self.asignee_id = row[3]
    self.issue_id = row[4]
    self.pull_request = row[5]
    self.pull_request_id = row[6]
    self.created_at = row[7]
    self.ext_ref_id = row[8]
    

def main():
  with open('conf/mysql.json', 'r') as f_in:
    conf = json.load(f_in)
  client = MysqlApi(conf)
  projects = client.get_projects()
  print(len(projects))
  print(len(client.get_watchers(projects[0])))

if __name__ == '__main__':
  main()