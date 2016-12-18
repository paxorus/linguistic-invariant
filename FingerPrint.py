#!/usr/bin/env python

"""
FingerPrint.py: training/testing layer for the probabilistic models

Usage:
	from FingerPrint import FingerPrint
	fp = FingerPrint()
	fp.train("hillary_train", "trump_train")
	fp.test("hillary_test", "trump_test")
	fp.print_stats()
"""

__author__ = "Prakhar Sahay"
__date__ = "December 16th, 2016"

# import statements
# from VanillaModel import VanillaModel as Model
# from BigramModel import BigramModel as Model
# from PosModel import PosModel as Model
from NaiveBayesModel import NaiveBayesModel as Model

class FingerPrint():

	def __init__(self):
		self.model = None

	def train(self, pos_file_name, neg_file_name):

		pos_file = open(pos_file_name, encoding="utf8")
		neg_file = open(neg_file_name, encoding="utf8")

		if self.model:
			self.model[0].analyze(pos_file)
			self.model[1].analyze(neg_file)
		else:
			self.model = (Model(pos_file), Model(neg_file))

	def test(self, pos_file_name, neg_file_name):

		(true_pos, pos) = self.classify(self.model[0], self.model[1], pos_file_name)
		(true_neg, neg) = self.classify(self.model[1], self.model[0], neg_file_name)
		false_pos = neg - true_neg

		# calculate statistics
		recall = true_pos / pos * 100
		precision = true_pos / (true_pos + false_pos) * 100
		accuracy = (true_pos + true_neg) / (pos + neg) * 100
		f_measure = 2 / (1/precision + 1/recall)

		self.stats = (recall, precision, accuracy, f_measure)

	def classify(self, pos_model, neg_model, test_file_name):

		pos_score = 0
		num_tests = 0
		file = open(test_file_name, encoding="utf8")

		# each line/dialog is a test
		for sample in file:

			score = pos_model.score(neg_model, sample)

			# if pos_model wins, increment pos_score
			if score > 0:
				pos_score += 1
			elif score == 0:
				pos_score += 0.5
			num_tests += 1

		return (pos_score, num_tests)

	def print_stats(self):
		stats = self.stats
		print("Recall: %.3f%%" % stats[0])
		print("Precision: %.3f%%" % stats[1])
		print("Accuracy: %.3f%%" % stats[2])
		print("F-measure: %.3f%%" % stats[3])
