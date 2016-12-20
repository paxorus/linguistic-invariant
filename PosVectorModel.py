#!/usr/bin/env python

"""
PosVectorModel.py: (part of speech, spelling) probability model

Usage:
	from PosVectorModel import PosVectorModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "December 19th, 2016"

# import statements
import nltk
import numpy as np
from operator import itemgetter

class PosVectorModel():

	def __init__(self, file):
		self.word_transition_cfd = nltk.ConditionalFreqDist()
		self.pos_tag_cfd = nltk.ConditionalFreqDist()
		self.tag_set = set()
		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		bigrams = []
		tag_lookup = []

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):
				words = nltk.word_tokenize(sentence)
				bigrams.extend(nltk.bigrams(words))
				tag_lookup.extend(nltk.pos_tag(words))

		# update distributions
		self.word_transition_cfd += nltk.ConditionalFreqDist(bigrams)# word -> next word
		self.pos_tag_cfd += nltk.ConditionalFreqDist(tag_lookup)# word -> POS tag

		# create an index mapping POS -> unique integer, for vector creation
		index = {}
		self.tag_set.update(map(itemgetter(1), tag_lookup))
		pos_tags = list(self.tag_set)# creates some arbitrary POS order

		for i in range(len(pos_tags)):
			index[pos_tags[i]] = i

		self.pos_index = index

	# give FreqDist of POS tags as a NumPy vector
	def pos_vector(self, word):
		pos_fd = self.pos_tag_cfd[word]
		vector = self.empty_pos_vector()

		for pos_tag in pos_fd.keys():
			idx = self.pos_index[pos_tag]
			vector[idx] = pos_fd.freq(pos_tag)

		return vector

	def empty_pos_vector(self):
		return np.zeros(len(self.pos_tag_cfd))

	# return positive number if self is better match for sample text
	# return negative number if other model is better match
	def score(self, other, sample):
		pos_score = 0
		neg_score = 0
		for sentence in nltk.sent_tokenize(sample):
			words = nltk.word_tokenize(sentence)
			pos_score += self.count(words)
			neg_score += other.count(words)

		return pos_score - neg_score

	def count(self, words):
		score = 0

		# walk num_words-1 times
		for idx in range(1, len(words)):
			fd = self.word_transition_cfd[words[idx - 1]]# most likely words to follow the previous word
			score += self.calculate_score(fd.keys(), words[idx])

		return score

	# assess how well actual_word fits FreqDist
	def calculate_score(self, predicted, actual_word):
		
		if not predicted:
			return 0

		if actual_word in predicted:
			return 1
		
		# attempt to assign partial credit with cosine similarity
		actual_vector = self.pos_vector(actual_word)
		predicted_vector = self.empty_pos_vector()
		for word in predicted:
			predicted_vector += self.pos_vector(word)
		
		predicted_vector /= len(predicted)

		return actual_vector.dot(predicted_vector)
