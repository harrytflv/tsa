import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from pivotal_tracker_api.tracker_api import TrackerApi

api = TrackerApi('02dd7e25c04dc4e72699947eaf9b25d7')

transition_time = [[], [], []]
for story in api.get_stories(1544059):
	if 'estimate' in story:
		estimate = story['estimate']
	else:
		estimate = 1
	started, finished, accepted = None, None, None
	for transition in api.get_story_transitions(1544059, story['id']):
		if transition['state'] == 'started':
			started = datetime.strptime(transition['occurred_at'][:19], '%Y-%m-%dT%H:%M:%S')
		elif transition['state'] == 'finished':
			finished = datetime.strptime(transition['occurred_at'][:19], '%Y-%m-%dT%H:%M:%S')
		elif transition['state'] == 'accepted':
			accepted = datetime.strptime(transition['occurred_at'][:19], '%Y-%m-%dT%H:%M:%S')
	if started and finished and accepted:
		transition_time[estimate-1].append((finished - started, accepted - finished))

fig, ax = plt.subplots()
plot_data = [[np.log(item[0].seconds / 3600.) for item in transition] for transition in transition_time]
plot_data.extend([[np.log(item[1].seconds / 3600.) for item in transition] for transition in transition_time])
plt.boxplot(plot_data)
plt.savefig('boxplot_collegeTrackMail.png')
plt.close(fig)
