from __future__ import division
import nltk
import operator
import math
import argparse
from random import randint, uniform
from collections import defaultdict
from sys import exit

class Quadragram:
    def __init__(self, tokens = None):
        if tokens:
            self.tokens = tokens
            self.total_words = len(tokens)
            self.trigrams = nltk.ngrams(tokens,3)
            self.quadragrams = nltk.ngrams(tokens,4)
        else:
            self.quadragram_dict = defaultdict(lambda: defaultdict(int))
            self.words = []
            self.sentence = []
            self.sentence_length = 20

    def generate_model(self):
        total_tokens = len(self.tokens)

        quadragram_count = defaultdict(int)
        trigram_count = defaultdict(int)


        for trigram in self.trigrams:
            trigram_count[trigram] += 1


        for quadragram in self.quadragrams:
            quadragram_count[quadragram] += 1


        rows_to_print = []

        for quadragram in quadragram_count:
            tmp = ""
            for entry in quadragram:
                tmp += entry + " "

            log_prob = "{0:.15f}".format(math.log(quadragram_count[quadragram] / trigram_count[quadragram[:3]]))

            rows_to_print.append(tmp.strip()  + " " + str(log_prob))

        return rows_to_print



    def read_quadragram_model(self, model):
        for row in model:
            quadragram = row.split(' ')
            if(len(quadragram) == 5):
                self.quadragram_dict[quadragram[0] + ' ' + quadragram[1] + ' ' + quadragram[2]][quadragram[1] + ' ' + quadragram[2] + ' ' + quadragram[3]] = math.exp(float(quadragram[4]))
        self.words = list(self.quadragram_dict.keys())
        self.generate_sentence(None,0)
        print(' '.join(self.sentence))

    def generate_sentence(self, first_token, iterations):
        if iterations == self.sentence_length:
            return

        if(iterations == 0):
            first_token = self.words[randint(0, len(self.quadragram_dict.keys()))]
            self.sentence.append(first_token)
            return self.generate_sentence(first_token, 1)

        quadragrams_for_word = list(self.quadragram_dict[first_token].items())


        probs = []
        for tuple in quadragrams_for_word:
            prob = float(tuple[1])
            probs.append(math.exp(prob))
        prob_sum = sum(probs)

        probs = [p/prob_sum for p in probs]

        acc = 0
        prob_array = []
        for p in probs:
            acc += p
            prob_array.append(acc)

        rand_index = uniform(0,1)
        next_token = ''
        for index, value in enumerate(prob_array):
            if rand_index < value:
                next_token = quadragrams_for_word[index][0]
                next_word = next_token.split(' ')[2]
                break
        self.sentence.append(next_word)
        iterations += 1
        return self.generate_sentence(next_token, iterations)
