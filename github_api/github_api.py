#!/usr/bin/env 

import requests
import json

class GithubApi(object):
  API_BASE = 'https://api.github.com'
  """docstring for GithubApi"""
  def __init__(self, token):
    super(GithubApi, self).__init__()
    self.token = token

  def get_commits(self, owner, repo, **params):
    url = self._resource('/repos/{owner}/{repo}/commits'.format(owner=owner, repo=repo))
    pages, links = self._request(url, requests.get, params)
    while 'next' in links:
      url = links['next']['url']
      page, links = self._request(url, requests.get, params)
      pages.extend(page)
    return pages

  def get_commit(self, owner, repo, sha):
    url = self._resource('/repos/{owner}/{repo}/commits/{sha}'.format(owner=owner, repo=repo, sha=sha))
    page, _ = self._request(url, requests.get)
    return page

  ''' Collaborators can only be accessed by a member of the repo. '''
  def get_collaborators(self, owner, repo):
    url = self._resource('/repos/{owner}/{repo}/collaborators'.format(owner=owner, repo=repo))
    page, _ = self._request(url, requests.get)
    return page

  ''' This only returns stats within a period '''
  def get_contributes(self, owner, repo):
    url = self._resource('/repos/{owner}/{repo}/stats/contributors'.format(owner=owner, repo=repo))
    page, _ = self._request(url, requests.get)
    return page

  def _add_headers(self, params={}):
    media = params.pop('media', 'application/vnd.github.v3+json')
    header = {
      'Accept':media,
      'Authorization':'token {}'.format(self.token)
    }
    header.update(params)
    return header

  def _request(self, url, method, params={}):
    resp = method(url, headers=self._add_headers(params))
    return resp.json(), resp.links

  def _resource(self, resource):
    return self.API_BASE + resource

def main():
  with open('../conf/tokens.json', 'r') as f_in:
    token = json.load(f_in)
  client = GithubApi(token['github']['token'])
  resp = client.get_commits('an-ju', 'projectscope')
  contributes = client.get_contributes('an-ju', 'projectscope')

if __name__ == '__main__':
  main()