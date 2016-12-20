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

class RandomWalker2():

	def __init__(self, model, words):
		self.model = model
		self.words = words
		self.score = 0

		# walk num_words-1 times
		for idx in range(1, len(words)):
			self.walk_to(idx)

	# predict word[idx], keep most frequently suggested
	def walk_to(self, idx):

		# use previous word to predict
		fd = self.model.word_transition_cfd[self.words[idx - 1]]# most likely words to follow the previous word
		actual_word = self.words[idx]
		self.score += self.calculate_score(fd, actual_word)

	# assess how well actual_word fits FreqDist
	def calculate_score(self, fd, actual_word):
		# examine top predictions
		pairs = fd.most_common(16)# [(a,20), (her,15), (my,10), (their,5)]
		predicted = get_column(pairs, 0)# [a, her, my, their]
		predicted = fd.keys()

		if not predicted:
			return 0

		if actual_word in predicted:
			idx = predicted.index(actual_word)
			self.score += pairs[idx][1] # count of actual_word in predicted
			return 1.5
		
		# attempt to assign partial credit
		# actual_vector = self.model.pos_vector(actual_word)
		# predicted_vector = self.model.empty_pos_vector()
		# for word in predicted:
		# 	predicted_vector += self.model.pos_vector(word)
		
		# predicted_vector /= len(predicted)

		# return actual_vector.dot(predicted_vector)
		return 0


def get_column(list_of_tuples, i):
	return list(map(itemgetter(i), list_of_tuples))