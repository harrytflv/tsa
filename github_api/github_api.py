#!/usr/bin/env 

import requests
import json

class GithubApi(object):
  API_BASE = 'https://api.github.com'
  """
    Wrapper for GitHub API access.

    WARNING: NO ERROR HANDLING! BREAK WHEN RETURN STATUS IF HTTP RESPONSE CODE IS AN ERROR.
  """
  def __init__(self, token):
    """Token is used to access GitHub."""
    super(GithubApi, self).__init__()
    self.token = token

  def get_commits(self, owner, repo, **params):
    """
      Get all git commits for a given repository.
      https://developer.github.com/v3/repos/commits/#list-commits-on-a-repository

      Input
        - owner, repo: used for locating the repository
        - params: other parameters, parameters will be included as headers
      Output
        - pages: a list of json objects (dictionaries), each dictionary is a commit.
      Note:
        Github divides results into pages, so this single function call may include several API calls
        see https://developer.github.com/guides/traversing-with-pagination/
    """
    url = self._resource('/repos/{owner}/{repo}/commits'.format(owner=owner, repo=repo))
    pages, links = self._request(url, requests.get, params)
    while 'next' in links:
      url = links['next']['url']
      page, links = self._request(url, requests.get, params)
      pages.extend(page)
    return pages

  def get_commit(self, owner, repo, sha):
    """
      Get a single commit
      https://developer.github.com/v3/repos/commits/#get-a-single-commit

      Input
        - owner, repo: used for locating the repository
        - sha: the sha for the commit
      Output
        - A json object (dictionary)
    """
    url = self._resource('/repos/{owner}/{repo}/commits/{sha}'.format(owner=owner, repo=repo, sha=sha))
    page, _ = self._request(url, requests.get)
    return page

  def get_collaborators(self, owner, repo):
    '''
      WARNING: Collaborators can only be accessed by a member of the repo.
      WARNING: NEED REFACTOR!

      Get the list of collaborators of a repository
      https://developer.github.com/v3/repos/#list-contributors

      Input
        - onwer, repo: used for locating the repository
      Output
        - A list of json objects, each is a collaborator
    '''
    url = self._resource('/repos/{owner}/{repo}/collaborators'.format(owner=owner, repo=repo))
    page, _ = self._request(url, requests.get)
    return page

  def get_contributes(self, owner, repo):
    '''
      WARNING: This only returns stats within a period

      Get contributors statistics of a repository
      https://developer.github.com/v3/repos/statistics/#get-contributors-list-with-additions-deletions-and-commit-counts

      Input
        - owner, repo: used for locating the repository
      Output
        - list of json objects, each is a contributor
    '''
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