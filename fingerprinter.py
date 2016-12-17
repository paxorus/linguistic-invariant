#!/usr/bin/env python

"""
fingerprinter.py: proof of concept script

Usage:
	python fingerprinter.py
"""

__author__ = "Prakhar Sahay"
__date__ = "November 10th, 2016"

from FingerPrint import FingerPrint

def main():
	fp = FingerPrint()
	fp.train("data/hillary-first-debate.txt", "data/trump-first-debate.txt")

	fp.train("data/hillary-second-debate.txt", "data/trump-second-debate.txt")

	fp.test("data/hillary-third-debate.txt", "data/trump-third-debate.txt") 
	fp.print_stats()

main()