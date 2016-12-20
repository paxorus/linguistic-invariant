#!/usr/bin/env python
#!/usr/bin/env python

"""
SimpleMatcherModel.py: (part of speech, spelling) probability model

Usage:
	from SimpleMatcherModel import SimpleMatcherModel as Model
	m = Model("hillary_train")
	m2 = Model("trump_train")
	score = m.score(m2, test_file.read())
"""

__author__ = "Prakhar Sahay"
__date__ = "December 19th, 2016"

# import statements
import nltk

class SimpleMatcherModel():

	def __init__(self, file):
		self.word_transition_cfd = nltk.ConditionalFreqDist()
		self.analyze(file)

	# build a probability model from the training file
	def analyze(self, file):

		bigrams = []

		# analyze each line as separate sample
		for sample in file:
			for sentence in nltk.sent_tokenize(sample):
				words = nltk.word_tokenize(sentence)
				bigrams.extend(nltk.bigrams(words))

		# update distributions
		self.word_transition_cfd += nltk.ConditionalFreqDist(bigrams)# word -> next word

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
		# walk num_words-1 times
		score = 0
		for idx in range(1, len(words)):
			fd = self.word_transition_cfd[words[idx - 1]]# most likely words to follow the previous word
			score += words[idx] in fd.keys()
		return score