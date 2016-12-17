#!/usr/bin/env python

"""
VanillaModel.py: unigram probability model, discards stopwords and non-alpha tokens

Usage:
	from VanillaModel import VanillaModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "November 10th, 2016"

# import statements
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords

class VanillaModel():

	stop_words = stopwords.words('english')

	def __init__(self, filename):
		self.avg_word_length = []
		self.lex_diversity = []
		self.nonstop = []
		self.fd = nltk.FreqDist()

		self.analyze(filename)

	# build a probability model from the training file
	def analyze(self, filename):
		# print(filename)
		file = open(filename, encoding="utf8")

		# analyze each line as separate sample
		for sample in file:
			num_words = 0
			total_length = 0
			vocab = set()

			for sentence in nltk.sent_tokenize(sample):
				for word in nltk.word_tokenize(sentence):

					num_words += 1
					total_length += len(word)

					vocab.add(word)
					if not self.is_significant(word):
						continue

					self.nonstop.append(word)

			self.avg_word_length.append(total_length / num_words)
			self.lex_diversity.append(num_words / len(vocab))

		self.fd = nltk.FreqDist(self.nonstop)

	# return positive integer if self is better match for sample text
	# return negative integer if other model is better match
	def score(self, other, sample):
		score = 0
		for sentence in nltk.sent_tokenize(sample):
			for word in nltk.word_tokenize(sentence):
				# if word in either model, lean towards higher frequency
				pos = self.frequency_of(word)
				neg = other.frequency_of(word)
				if pos > neg:
					score += 1
				elif pos < neg:
					score -= 1
		return score

	# helper
	def frequency_of(self, word):
		return self.fd.get(word, 0) / self.fd.N()

	# helper
	def is_significant(self, word):
		return word not in self.stop_words and "'" not in word and word.isalpha()
		# eliminate 's 're n't 've
		# no numbers or punctuation

	## STYLOMETRICS ##

	# lexical diversity
	def get_lex_diversity(self):
		return self.vector_dist(self.lex_diversity)

	# average word length
	def get_avg_word_length(self):
		return self.vector_dist(self.avg_word_length)

	# most common significant words
	def get_most_common(self, n=10):
		return self.fd.most_common(n)

	# helper
	def vector_dist(self, vector):
		if not vector:
			return []

		vector = sorted(vector)
		vlen = len(vector)
		quartiles = []
		for i in range(4):
			quartile = vector[math.floor(i / 4 * vlen)]
			quartiles.append(quartile)
		quartiles.append(vector[-1])
		return quartiles

	# words appearing more in this distribution, most to least common
	def diff_most_common(self, other, n=10):
		diff_fd = {}
		for word in self.fd.keys():
			if word in other.fd:
				diff_fd[word] = self.fd[word] / self.fd.N() - other.fd[word] / other.fd.N()

		lookup = lambda w: diff_fd[w]
		most_common = sorted(diff_fd.keys(), key=lookup)
		# most_common = sorted(diff_fd.items(), key=itemgetter(1))
		return [most_common[-n:], most_common[:n]]

	# words appearing only in this distribution, most to least common
	def unique_to_say(self, other, n=0):
		unique_words = []
		for word in self.fd.keys():
			if word not in other.fd:
				unique_words.append((word, self.fd[word]))

		lookup = lambda w: self.fd[w]
		# unique_words = sorted(unique_words, key=lookup)
		unique_words = sorted(unique_words, key=itemgetter(1), reverse=True)

		if n < 1:
			n = len(unique_words)
		return unique_words[:n]
