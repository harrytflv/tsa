#!/usr/bin/env 

import urllib.request
import json

class TrackerApi(object):
	"""
		Wrapper for PivotalTracker API

    WARNING: NO ERROR HANDLING! BREAK WHEN RETURN STATUS IF HTTP RESPONSE CODE IS AN ERROR.
	"""
	API_BASE = 'https://www.pivotaltracker.com/services/v5/'
	def __init__(self, token):
		"""Token is used for accessing projects"""
		self.token = token

	def get_story_transitions(self, project_id, story_id):
		"""
			Get all transitions of a user story
			https://www.pivotaltracker.com/help/api/rest/v5#Story_Transitions

			Input
				- project_id: Project ID
				- story_id: Story ID
			Output
				- a list of json objects, each is a story transition
		"""

		url = self._resource('projects/{}/stories/{}/transitions'.format(project_id, story_id))
		return self._request(url)

	def get_proejct(self, project_id):
		"""
			List a given project
			https://www.pivotaltracker.com/help/api/rest/v5#Project

			Input
				- project_id: Project ID
			Output
				- a json object
		"""
		url = self._resource('projects/{}'.format(project_id))
		return self._request(url)

	def get_stories(self, project_id):
		"""
			Fetch all stories of a project
			https://www.pivotaltracker.com/help/api/rest/v5#Stories

			Input
				- project_id: Project ID
			Output
				- a list of json objects, each object is a story
		"""
		url = self._resource('projects/{}/stories'.format(project_id))
		return self._request(url)

	def get_account(self, account_id):
		"""
			Access a account
			https://www.pivotaltracker.com/help/api/rest/v5#Account

			Input
				- account_id: Account ID
			Output
				- a json object
		"""
		url = self._resource('accounts/{}'.format(account_id))
		return self._request(url)

	def get_story_owners(self, project_id, story_id):
		"""
			Get a list of onwers of a story
			https://www.pivotaltracker.com/help/api/rest/v5#projects_project_id_stories_story_id_owners_get

			Input
				- project_id: Project ID
				- story_id: Story ID
			Output
				- a list of json objects, each is an owner
		"""
		url = self._resource('projects/{project_id}/stories/{story_id}/owners'.format(project_id=project_id, story_id=story_id))
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
	print(client.get_story_owners('934278', '60145984'))

if __name__ == '__main__':
	main()