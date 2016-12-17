#!/usr/bin/env python

"""
PosModel.py: (part of speech, spelling) probability model

Usage:
	from PosModel import PosModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "December 16th, 2016"

# import statements
import nltk

class PosModel():


	def __init__(self, file):
		self.tagged = []
		# self.fd = nltk.FreqDist()

		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):
				words = nltk.word_tokenize(sentence)
				tagged = nltk.pos_tag(words)
				self.tagged.extend(tagged)

		self.pair_fd = nltk.FreqDist(self.tagged)
		words, pos_tags = zip(*self.tagged)
		self.word_fd = nltk.FreqDist(words)
		self.tag_fd = nltk.FreqDist(pos_tags)

	# return positive integer if self is better match for sample text
	# return negative integer if other model is better match
	def score(self, other, sample):

		score = 0
		for sentence in nltk.sent_tokenize(sample):
			for (word, pos_tag) in nltk.pos_tag(nltk.word_tokenize(sentence)):
				# if word in either model, lean towards higher frequency
				pos = self.frequency_of(other, word, pos_tag)
				neg = other.frequency_of(self, word, pos_tag)

				if pos > neg:
					score += 1
				elif pos < neg:
					score -= 1
		return score

	# helper
	def frequency_of(self, other, word, pos_tag):

		pair = (word, pos_tag)
		if pair in self.pair_fd:
			# P(word+pos | self) / P(word+pos)
			likelihood = self.pair_fd[pair] / self.pair_fd.N()
			predictor_prior = (self.pair_fd[pair] + other.pair_fd[pair]) / (self.pair_fd.N() + other.pair_fd.N())
			return  likelihood / predictor_prior

		if pos_tag in self.tag_fd:
			# P(tag | self) / P(tag)
			likelihood = self.tag_fd[pos_tag] / self.tag_fd.N()
			predictor_prior = (self.tag_fd[pos_tag] + other.tag_fd[pos_tag]) / (self.tag_fd.N() + other.tag_fd.N())
			return likelihood / predictor_prior

		return 0

	# helper
	def is_significant(self, word):
		return "'" not in word
		# eliminate 's 're n't 've
