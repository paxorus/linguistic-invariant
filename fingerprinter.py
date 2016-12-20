#!/usr/bin/env python

"""
fingerprinter.py: top-level script

Usage:
	python fingerprinter.py
"""

__author__ = "Prakhar Sahay"
__date__ = "November 10th, 2016"

import time
from FingerPrint import FingerPrint

def main():
	fp = FingerPrint()
	t0 = time.time()

	fp.train("data/hillary-first-debate.txt", "data/trump-first-debate.txt")
	fp.train("data/hillary-second-debate.txt", "data/trump-second-debate.txt")
	t1 = time.time()

	fp.test("data/hillary-third-debate.txt", "data/trump-third-debate.txt")
	t2 = time.time()

	fp.print_stats()
	print("Training time: %.3fs" % (t1 - t0))
	print("Testing time: %.3fs" % (t2 - t1))
main()