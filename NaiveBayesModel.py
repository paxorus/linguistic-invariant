#!/usr/bin/env python

"""
NaiveBayesModel.py: (part of speech, spelling) probability model

Usage:
	from NaiveBayesModel import NaiveBayesModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "December 17th, 2016"

# import statements
import nltk
from operator import itemgetter

class NaiveBayesModel():

	def __init__(self, file):
		self.tagged = []# [(DT, The), (NN, dog), ...]
		self.transitions = []# [(DT, NN), ...]
		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):
				words = nltk.word_tokenize(sentence)# [The, dog, ...]
				tagged = nltk.pos_tag(words)# [(The, DT), (dog, NN), ...]
				# better performance if tag is condition and word is sample
				tagged2 = [(tag, word) for (word, tag) in tagged]# [(DT, The), (NN, dog), ...]
				self.tagged.extend(tagged2)
				tag_sequence = list(map(itemgetter(1), tagged))# [DT, NN, ...]
				self.transitions.extend(nltk.bigrams(tag_sequence))

		# update distributions
		self.pair_cfd = nltk.ConditionalFreqDist(self.tagged)
		self.transition_cfd = nltk.ConditionalFreqDist(self.transitions)

	# return positive number if self is better match for sample text
	# return negative number if other model is better match
	def score(self, other, sample):

		pos_score = 1
		neg_score = 1
		for sentence in nltk.sent_tokenize(sample):
			tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
			for (wtp1, wtp2) in nltk.bigrams(tagged):
				# twp: (word, tag) pair
				pos_score *= self.frequency_of(other, wtp1, wtp2)
				neg_score *= other.frequency_of(self, wtp1, wtp2)

			# get output probability of the first (word, tag) pair also
			first_pair = tagged[0]
			pos_score *= self.p_output(other, first_pair[1], first_pair[0])
			neg_score *= other.p_output(self, first_pair[1], first_pair[0])

		return pos_score - neg_score

	# helper
	def frequency_of(self, other, wtp1, wtp2):

		# unpack tagged bigram
		tag1 = wtp1[1]
		word2,tag2 = wtp2

		p_transition = self.p_transition(other, tag1, tag2)
		p_output = self.p_output(other, tag2, word2)

		return p_output * p_transition

	# computes P(t2+t1 | self) / P(t2+t1)
	def p_transition(self, other, tag1, tag2):
		if tag1 in self.transition_cfd:
			fd = self.transition_cfd[tag1]
			if tag1 in other.transition_cfd:
				other_fd = other.transition_cfd[tag1]
			else:
				other_fd = nltk.FreqDist()

			if tag2 in fd:
				likelihood = fd[tag2] / fd.N()
				predictor_prior = (fd[tag2] + other_fd[tag2]) / (fd.N() + other_fd.N())
				return likelihood / predictor_prior

		return 1 / self.transition_cfd.N()


	# computes P(w+t | self) / P(w+t)
	def p_output(self, other, tag, word):
		if tag in self.pair_cfd:
			fd = self.pair_cfd[tag]
			if tag in other.pair_cfd:
				other_fd = other.pair_cfd[tag]
			else:
				other_fd = nltk.FreqDist()

			if word in fd:
				likelihood = fd[word] / fd.N()
				predictor_prior = (fd[word] + other_fd[word]) / (fd.N() + other_fd.N())
				return likelihood / predictor_prior

		return 1 / self.pair_cfd.N()