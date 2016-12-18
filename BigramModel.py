#!/usr/bin/env python

"""
BigramModel.py: bigram probability model

Usage:
	from BigramModel import BigramModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "December 16th, 2016"

# import statements
import nltk

class BigramModel():

	def __init__(self, file):
		self.words = []
		self.bigrams = []
		self.fd = nltk.FreqDist()
		self.cfd = nltk.ConditionalFreqDist()

		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):

				words = []
				for word in nltk.word_tokenize(sentence):
					if self.is_significant(word):
						words.append(word)

				self.words.extend(words)
				self.bigrams.extend(nltk.bigrams(words))

		self.fd = nltk.FreqDist(self.words)
		self.cfd = nltk.ConditionalFreqDist(self.bigrams)


	# return positive integer if self is better match for sample text
	# return negative integer if other model is better match
	def score(self, other, sample):
		score = 0
		for sentence in nltk.sent_tokenize(sample):
			for (w1, w2) in nltk.bigrams(nltk.word_tokenize(sentence)):

				# if word in either model, lean towards higher frequency
				pos = self.frequency_of(w1, w2)
				neg = other.frequency_of(w1, w2)
				if pos > neg:
					score += 1
				elif pos < neg:
					score -= 1
		return score

	# helper
	def frequency_of(self, w1, w2):
		if w1 in self.cfd:
			# P(w2 | w1, self)
			fd = self.cfd[w1]
			if w2 in fd:
				return fd[w2] / fd.N()

		if w2 in self.fd:
			# P(w2 | self)
			return self.fd[w2] / self.fd.N()

		return 0

	# helper
	def is_significant(self, word):
		return "'" not in word
		# eliminate 's 're n't 've
