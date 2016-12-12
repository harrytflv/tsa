#!/usr/bin/env 

import urllib.request
import json

class TrackerApi:
	API_BASE = 'https://www.pivotaltracker.com/services/v5/'
	def __init__(self, token):
		self.token = token

	def get_story_transitions(self, project_id, story_id):
		url = self._resource('projects/{}/stories/{}/transitions'.format(project_id, story_id))
		return self._request(url)

	def get_proejct(self, project_id):
		url = self._resource('projects/{}'.format(project_id))
		return self._request(url)

	def get_stories(self, project_id):
		url = self._resource('projects/{}/stories'.format(project_id))
		return self._request(url)

	def _add_headers(self, req, params = {}):
		req.add_header('Content-Type', 'application/json')
		req.add_header('X-TrackerToken', self.token)
		for key, content in params:
			req.add_header(key, content)

	def _request(self, url, params={}):
		req = urllib.request.Request(url)
		self._add_headers(req, params)
		resp = urllib.request.urlopen(req)
		return json.loads(resp.read().decode('utf-8'))

	def _resource(self, resource):
		return self.API_BASE + resource

def main():
	with open('../conf/tokens.json', 'r') as f_in:
		tokens = json.load(f_in)
	client = TrackerApi(tokens['pivotal_tracker']['token'])
	# print(client.get_proejct('934278'))
	# print(client.get_story_transitions('934278', '60145984'))
	print(client.get_stories('934278'))

if __name__ == '__main__':
	main()