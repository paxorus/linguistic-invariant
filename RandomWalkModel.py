#!/usr/bin/env python

"""
RandomWalkModel.py: (part of speech, spelling) probability model

Usage:
	from RandomWalkModel import RandomWalkModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "December 19th, 2016"

# import statements
import nltk
from operator import itemgetter

class RandomWalkModel():

	def __init__(self, file):
		self.back2_cfd = nltk.ConditionalFreqDist()
		self.back1_cfd = nltk.ConditionalFreqDist()
		self.forward1_cfd = nltk.ConditionalFreqDist()
		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		back2, back1, forward1 = [], [], []

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):
				words = lower_tokenize(sentence)

				# collect the (-1) and (+1) contexts into a CFD
				bigrams = nltk.bigrams(words)
				back1.extend(bigrams)
				forward1.extend([(bigram[1], bigram[0]) for bigram in bigrams])

				# collect the (-2) context into a CFD
				back2.extend([(trigram[0], trigram[2]) for trigram in nltk.trigrams(words)])

		# update distributions
		self.back2_cfd += nltk.ConditionalFreqDist(back2)
		self.back1_cfd += nltk.ConditionalFreqDist(back1)
		self.forward1_cfd += nltk.ConditionalFreqDist(forward1)

	# return positive number if self is better match for sample text
	# return negative number if other model is better match
	def score(self, other, sample):

		pos_score = 0
		neg_score = 0
		for sentence in nltk.sent_tokenize(sample):
			words = lower_tokenize(sentence)
			pos_score += RandomWalker(self, words).score
			neg_score += RandomWalker(other, words).score

		return pos_score - neg_score

# non-deterministcally walk
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

		actual_word = self.words[idx]
		total_fd = sum(fds, nltk.FreqDist())
		average_fd = self.scaleFd(total_fd, 1/len(fds))
		self.states = self.getNextStates(average_fd, actual_word)

		# print(total_fd)
		# print(self.states)

	def scaleFd(self, fd, scalar):
		weightedMap = {}
		for next_state,freq in fd.most_common(self.MAX_STATES):# (that,30)
			weightedMap[next_state] = freq * scalar# {that:9 = 6/20 * 30}

		return nltk.FreqDist(weightedMap)

	# use FreqDist to pick most significant next states
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


def lower_tokenize(sentence):
	words = nltk.word_tokenize(sentence)# [The, dog, ...]
	return list(map(lambda w: w.lower(), words))

def get_column(list_of_tuples, i):
	return list(map(itemgetter(i), list_of_tuples))