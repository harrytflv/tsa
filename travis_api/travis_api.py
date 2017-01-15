#!/usr/bin/env 

import requests
import json

class TravisApi(object):
  """
    Wrapper for Travis CI API access.

    WARNING: NO ERROR HANDLING! BREAK WHEN RETURN STATUS IF HTTP RESPONSE CODE IS AN ERROR.
  """
  API_BASE = 'https://api.travis-ci.org/'
  def __init__(self, token):
    """
      Token is the access token of Travis CI. Right now it assumes the token won't expire.
    """
    super(TravisApi, self).__init__()
    self.token = token

  def get_account(self):
    """
      Get account information.
      https://docs.travis-ci.com/api#accounts
    """
    return self._request(self._resource('accounts'), requests.get)

  def list_builds(self, owner, repo, **params):
    """
      List all builds of a given repository
      The pagination is handled automatically, so it always returns all builds
      https://docs.travis-ci.com/api#builds

      Input
        - owner, repo: used to locate the repository
      Output
        - A list of builds, where each build is a dictionary returned by the API
    """
    def make_func(page_number):
      def get_with_params(url, headers):
        return requests.get(url, headers=headers, params={'after_number': page_number})
      return get_with_params

    url = self._resource('repos/{}/{}/builds'.format(owner, repo))
    builds = self._request(url, requests.get, params)['builds']
    least_number = min([int(bd['number']) for bd in builds])
    while least_number > 1:
      page = self._request(url, make_func(least_number), params)['builds']
      builds.extend(page)
      least_number = min([int(bd['number']) for bd in page])
    return builds

  def _add_headers(self, params={}):
    media = params.pop('media', 'application/vnd.travis-ci.2+json')
    user_agent = params.pop('User-Agent', 'Travis/1.6.8')
    content_type = params.pop('Content-Type', 'application/json')
    header = {
      'Accept':media,
      'Authorization':'token {}'.format(self.token),
      'User-Agent': user_agent,
      'Content-Type': content_type
    }
    header.update(params)
    return header

  def _request(self, url, method, params={}):
    resp = method(url, headers=self._add_headers(params))
    return resp.json()

  def _resource(self, resource):
    return self.API_BASE + resource

def main():
  with open('../conf/tokens.json', 'r') as f_in:
    tokens = json.load(f_in)
  client = TravisApi(tokens['travis']['token'])
  print(client.get_account())
  builds = client.list_builds('DrakeW', 'projectscope')
  print(len(builds))
  # print([build['number'] for build in builds])

if __name__ == '__main__':
  main()