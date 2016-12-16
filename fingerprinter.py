#!/usr/bin/env python

"""
fingerprinter.py: proof of concept script

Usage:
	python fingerprinter.py
"""

__author__ = "Prakhar Sahay"
__date__ = "November 10th, 2016"

from VanillaFingerPrint import FingerPrint
import random
import nltk

def main():

	trained = train("data/hillary-second-debate.txt", "data/trump-second-debate.txt")
	test_results = test(trained, "data/hillary-third-debate.txt", "data/trump-third-debate.txt") 
	print_stats(test_results)

def train(pos_file, neg_file):
	return (FingerPrint(pos_file), FingerPrint(neg_file))

def test(trained, pos_test, neg_test):
	pos_train = trained[0]
	neg_train = trained[1]

	(true_pos, pos) = classify(pos_train, neg_train, pos_test)
	(true_neg, neg) = classify(neg_train, pos_train, neg_test)
	false_pos = neg - true_neg

	recall = true_pos / pos * 100
	precision = true_pos / (true_pos + false_pos) * 100
	accuracy = (true_pos + true_neg) / (pos + neg) * 100
	f_measure = 2 / (1/precision + 1/recall)

	return (recall, precision, accuracy, f_measure)

def print_stats(results):
	print("Recall: %.3f%%" % results[0])
	print("Precision: %.3f%%" % results[1])
	print("Accuracy: %.3f%%" % results[2])
	print("F-measure: %.3f%%" % results[3])


def classify(pos_model, neg_model, test_file):

	pos_score = 0
	num_tests = 0
	file = open(test_file, encoding="utf8")

	# each line/dialog is a test
	for sample in file:
		miniscore = 0
		for sentence in nltk.sent_tokenize(sample):
			for word in nltk.word_tokenize(sentence):
				# if word in either model, lean towards higher frequency
				pos = pos_model.frequency_of(word)
				neg = neg_model.frequency_of(word)
				if pos > neg:
					miniscore += 1
				elif pos < neg:
					miniscore -= 1

		num_tests += 1
		# if pos_model wins, increments pos_score
		if miniscore > 0:
			pos_score += 1
		elif miniscore == 0:
			# pos_score += random.choice([0, 1])
			pos_score += 0.5

	return (pos_score, num_tests)

main()