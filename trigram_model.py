from __future__ import division
import nltk
import operator
import math
import argparse
from random import randint, uniform
from collections import defaultdict
from sys import exit

class Trigram:
    def __init__(self, tokens = None):
        if tokens:
            self.tokens = tokens
            self.total_words = len(tokens)
            self.bigrams = nltk.ngrams(tokens,2)
            #self.uniq_words = set()
            self.trigrams = nltk.ngrams(tokens,3)
        
        self.trigram_dict = defaultdict(lambda: defaultdict(int))
        self.words = []
        self.sentence = []
        self.sentence_length = 20

    def generate_model(self):
        total_tokens = len(self.tokens)

        trigram_count = defaultdict(int)
        bigram_count = defaultdict(int)

        # for token in self.tokens:
        #     self.uniq_words.add(token)
        #     self.unigram_count[token.strip()] += 1

        for bigram in self.bigrams:
            bigram_count[bigram] += 1



        for trigram in self.trigrams:
            trigram_count[trigram] += 1


        rows_to_print = []

        for trigram in trigram_count:
            tmp = ""
            for entry in trigram:
                tmp += entry + " "

            log_prob = "{0:.15f}".format(math.log(trigram_count[trigram] / bigram_count[trigram[:2]]))

            rows_to_print.append(tmp.strip()  + " " + str(log_prob))

        return rows_to_print



    def read_trigram_model(self, model):
        for row in model:
            trigram = row.split(' ')
            if(len(trigram) == 4):
                self.trigram_dict[trigram[0] + ' ' + trigram[1]][trigram[1] + ' ' + trigram[2]] = math.exp(float(trigram[3]))
        self.words = list(self.trigram_dict.keys())
        self.generate_sentence(None,0)
        print(' '.join(self.sentence))

    def generate_sentence(self, first_token, iterations):
        if iterations == self.sentence_length:
            return

        if(iterations == 0):
            first_token = self.words[randint(0, len(self.trigram_dict.keys()))]
            self.sentence.append(first_token)
            return self.generate_sentence(first_token, 1)

        trigrams_for_word = list(self.trigram_dict[first_token].items())


        probs = []
        for tuple in trigrams_for_word:
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
                next_token = trigrams_for_word[index][0]
                next_word = next_token.split(' ')[1]
                break
        self.sentence.append(next_word)
        iterations += 1
        return self.generate_sentence(next_token, iterations)
