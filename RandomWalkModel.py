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
__date__ = "December 17th, 2016"

# import statements
import nltk
from operator import itemgetter

class RandomWalkModel():

	def __init__(self, file):
		# self.tagged = []# [(DT, The), (NN, dog), ...]
		# self.transitions = []# [(DT, NN), ...]
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
				words = nltk.word_tokenize(sentence)# [The, dog, ...]
				words = list(map(lambda w: w.lower(), words))
				# tagged = nltk.pos_tag(words)# [(The, DT), (dog, NN), ...]
				# tag_sequence = list(map(itemgetter(1), tagged))# [DT, NN, ...]

				# collect the (-1) and (+1) contexts into a CFD
				bigrams = nltk.bigrams(words)
				back1.extend(bigrams)
				forward1.extend([(bigram[1], bigram[0]) for bigram in back1])

				# collect the (-2) context into a CFD
				back2.extend([(trigram[0], trigram[2]) for trigram in nltk.trigrams(words)])

		# update distributions
		self.back2_cfd += nltk.ConditionalFreqDist(back2)
		self.back1_cfd += nltk.ConditionalFreqDist(back1)
		self.forward1_cfd += nltk.ConditionalFreqDist(forward1)

	# return positive number if self is better match for sample text
	# return negative number if other model is better match
	def score(self, other, sample):

		pos_score = 1
		neg_score = 1
		for sentence in nltk.sent_tokenize(sample):
			words = nltk.word_tokenize(sentence)
			words = list(map(lambda w: w.lower(), words))

			# tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
			# tag_sequence = list(map(itemgetter(1), tagged))# [DT, NN, ...]

			pos_walker = RandomWalker(self, words)
			neg_walker = RandomWalker(other, words)

			# for idx in nltk.bigrams(tagged):

				# pos_score *= self.frequency_of(tag_sequence, wtp2)
				# neg_score *= other.frequency_of(wtp1, wtp2)

			# get output probability of the first (word, tag) pair also
			# first_pair = tagged[0]
			# pos_score *= self.p_output(other, first_pair[1], first_pair[0])
			# neg_score *= other.p_output(self, first_pair[1], first_pair[0])

		return pos_walker.score - neg_walker.score

	# helper
	# def frequency_of(self, other, wtp1, wtp2):

# non-deterministcally walk
class RandomWalker():

	MAX_STATES = 10

	def __init__(self, model, words):
		# self.states = [words[0]]
		# self.idx = 1
		self.model = model
		self.words = words
		self.score = 0

		# initialize states
		fd = self.model.forward1_cfd[words[1]]
		self.states = self.getNextStates(fd, words[0])

		# pairs = fd.most_common(self.MAX_STATES)# [(a, 20), (her, 10), (my, 6), (their, 1)]
		# similars = list(map(itemgetter(0), pairs))# [a, her, my, their]
		# print(similars)

		# if words[0] in similars:
		# 	idx = similars.index(words[0])
		# 	self.score += pairs[idx][1] # count of words[0] in similars
		# else:
		# 	if similars:
		# 		similars.pop()# [a, her, my]
		# 	similars = [words[0]] + similars# [the, a, her, my]

		# self.states = similars

		# walk num_words-1 times
		for idx in range(1, len(words)):
			self.walk_to(idx)

	def walk_to(self, idx):
		# previous
		total_fd = nltk.FreqDist()

		if idx >= 1:
			# use states as (-1) context
			for state in self.states:
				fd = self.model.back1_cfd[state]
				total_fd += fd

		if idx >= 2:
			# (-2) context
			fd = self.model.back2_cfd[self.words[idx - 2]]
			total_fd += fd

		if idx < len(self.words) - 1:
			fd = self.model.forward1_cfd[self.words[idx + 1]]
			total_fd += fd

		actual_word = self.words[idx]
		self.states = self.getNextStates(fd, actual_word)


	# use FreqDist to 
	# increase score if actual word 
	def getNextStates(self, fd, actual_word):
		pairs = fd.most_common(self.MAX_STATES)
		similars = list(map(itemgetter(0), pairs))# [a, her, my, their]
		print(similars)

		if actual_word in similars:
			idx = similars.index(actual_word)
			self.score += pairs[idx][1] # count of words[idx] in similars
		else:
			if similars:
				similars.pop()# [a, her, my]
			similars = [actual_word] + similars# [the, a, her, my]

		return similars
