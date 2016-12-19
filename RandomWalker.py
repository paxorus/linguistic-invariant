#!/usr/bin/env python

"""
RandomWalker.py: use surrounding textual context (previous two words and next word) and the previous set
of states to predict a new set of states. At any time, the states are words considered "similar" to the
current word. If the "similar" words contain the current word, the score is increased.

Usage:
	from RandomWalker import RandomWalker
	score = RandomWalker(randomWalkModel, words).score
"""

import nltk
from operator import itemgetter

class RandomWalker():

	MAX_STATES = 7

	def __init__(self, model, words):
		self.model = model
		self.words = words
		self.score = 0
		self.states = []

		# walk num_words-1 times
		for idx in range(1, len(words)):
			self.walk_to(idx)

	# predict word[idx], keep most frequently suggested
	def walk_to(self, idx):
		# total_fd = nltk.FreqDist()
		fds = []

		# use states as (-1) context
		# each state should not count toward total_fd equally, weigh with /num_states
		state_avg_fd = nltk.FreqDist()
		total_weight = sum(map(itemgetter(1), self.states))# 20
		for state,weight in self.states:# (is,6)
			fd = self.model.back1_cfd[state]# {that:30, ...}
			state_avg_fd += self.scaleFd(fd, weight / total_weight)
		if total_weight:
			fds.append(state_avg_fd)

		# (-1) context
		fd = self.model.back1_cfd[self.words[idx - 1]]	
		fds.append(fd)

		# (-2) context
		if idx >= 2:
			fd = self.model.back2_cfd[self.words[idx - 2]]
			fds.append(fd)

		# (+1) context
		if idx < len(self.words) - 1:
			fd = self.model.forward1_cfd[self.words[idx + 1]]
			fds.append(fd)

		# (+2) context
		if idx < len(self.words) - 2:
			fd = self.model.forward2_cfd[self.words[idx + 2]]
			fds.append(fd)

		# average the above information 
		actual_word = self.words[idx]
		total_fd = sum(fds, nltk.FreqDist())
		average_fd = self.scaleFd(total_fd, 1/len(fds))
		self.states = self.getNextStates(average_fd, actual_word)

	def scaleFd(self, fd, scalar):
		weightedMap = {}
		for next_state,freq in fd.most_common(self.MAX_STATES):# (that,30)
			weightedMap[next_state] = freq * scalar# {that:9 = 6/20 * 30}

		return nltk.FreqDist(weightedMap)

	# use FreqDist to predict next states
	# increase score if actual word 
	def getNextStates(self, fd, actual_word):
		pairs = fd.most_common(self.MAX_STATES + 1)# [(a,20), (her,15), (my,10), (their,5)]
		similars = get_column(pairs, 0)# [a, her, my, their]

		if actual_word in similars:
			idx = similars.index(actual_word)
			self.score += pairs[idx][1] # count of actual_word in similars
			del pairs[idx]
		elif pairs:
			pairs.pop()

		return pairs


def get_column(list_of_tuples, i):
	return list(map(itemgetter(i), list_of_tuples))