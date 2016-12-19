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
from RandomWalker import RandomWalker

class RandomWalkModel():

	def __init__(self, file):
		self.back2_cfd = nltk.ConditionalFreqDist()
		self.back1_cfd = nltk.ConditionalFreqDist()
		self.forward1_cfd = nltk.ConditionalFreqDist()
		self.forward2_cfd = nltk.ConditionalFreqDist()
		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		back2, back1, forward1, forward2 = [], [], [], []

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):
				words = lower_tokenize(sentence)

				# collect the (-1) and (+1) contexts into a CFD
				bigrams = nltk.bigrams(words)
				back1.extend(bigrams)
				forward1.extend([(bigram[1], bigram[0]) for bigram in bigrams])

				# collect the (-2) context into a CFD
				trigrams = nltk.trigrams(words)
				back2.extend([(trigram[0], trigram[2]) for trigram in trigrams])
				forward2.extend([(trigram[2], trigram[0]) for trigram in trigrams])

		# update distributions
		self.back2_cfd += nltk.ConditionalFreqDist(back2)
		self.back1_cfd += nltk.ConditionalFreqDist(back1)
		self.forward1_cfd += nltk.ConditionalFreqDist(forward1)
		self.forward2_cfd += nltk.ConditionalFreqDist(forward2)

	# return positive number if self is better match for sample text
	# return negative number if other model is better match
	def score(self, other, sample):
		pos_score = 0
		neg_score = 0
		for sentence in nltk.sent_tokenize(sample):
			words = lower_tokenize(sentence)
			pos_score += RandomWalker(self, words).score
			neg_score += RandomWalker(other, words).score

		print(pos_score, neg_score)
		return pos_score - neg_score

def lower_tokenize(sentence):
	words = nltk.word_tokenize(sentence)# [The, dog, ...]
	return list(map(lambda w: w.lower(), words))
