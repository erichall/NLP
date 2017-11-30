from __future__ import division
from sys import exit
import nltk
import operator
import math
import argparse
from random import randint, uniform
from collections import defaultdict

class Ngram:
    def __init__(self, ngram, tokens = None):
        if tokens:
            self.tokens = tokens
            self.ngram = ngram 
            if ngram >= 1:
                self.unigrams = nltk.ngrams(tokens,1)
                self.unigram_count = defaultdict(int)
                self.get_unigram_count()
            if ngram >= 2:
                self.bigrams = nltk.ngrams(tokens,2)
                self.bigram_count = defaultdict(int)
                self.get_bigram_count()
            if ngram >= 3:
                self.trigrams = nltk.ngrams(tokens,3)    
                self.trigram_count = self.get_trigram_count()
            if ngram >= 4:
                self.quadragram = nltk.ngrams(tokens,4)    
                self.quadragram_count = self.get_quadragram_count() 
        else:
            if ngram >= 1:
                self.unigram_dict = defaultdict(int)
            if ngram >= 2:
                self.bigram_dict = defaultdict(lambda: defaultdict(int))
            if ngram >= 3:
                self.trigram_dict = defaultdict(lambda: defaultdict(int))
            if ngram >= 4:
                self.quadragram_dict = defaultdict(lambda: defaultdict(int))

            self.words = [] 
            self.sentence = []
            self.sentence_length = 20

    def generate_model(self): 
        if(self.ngram == 1):
            return self.unigram_probs()
        elif(self.ngram == 2):
            return self.bigram_probs()
        # elif(ngram == 3):

    def get_unigram_count(self):
        for token in self.unigrams:
            self.unigram_count[token[0].strip()] += 1
    def get_bigram_count(self):
        for bigram in self.bigrams:
            self.bigram_count[bigram] += 1
    def get_trigram_count(self):
        for trigram in self.trigrams:
            self.trigram_count[trigram] += 1
    def get_quadragram_count(self):
        for quad in self.quadragram:
            self.quadragram_count[quad] += 1
    
    def unigram_probs(self):
        all_unigrams = sum([word_count for word_count in self.unigram_count])
        rows_to_print = []
        for unigram in self.unigram_count:
            log_prob = "{0:.15f}".format(math.log(self.unigram_count[unigram] / all_unigrams))
            rows_to_print.append(unigram + ' ' + str(log_prob))
        return rows_to_print

    def bigram_probs(self): 
        rows_to_print = []

        for bigram in self.bigram_count:
            tmp = ""
            for entry in bigram:
                tmp += entry + " "
            log_prob = "{0:.15f}".format(math.log(self.bigram_count[bigram] / self.unigram_count[bigram[0]]))

            rows_to_print.append(tmp.strip()  + " " + str(log_prob))
        return rows_to_print
    
    def read_unigram_model(self, model):
        for row in model:
            unigram = row.split(' ')
            self.unigram_dict[unigram[0]] = math.exp(float(unigram[1]))
        self.words = list(self.unigram_dict.keys())
        self.generate_sentence(None, 0)
        print(' '.join(self.sentence))

    def read_bigram_model(self, model):
        for row in model:
            bigram = row.split(' ')
            if(len(bigram) == 3):
                self.bigram_dict[bigram[0]][bigram[1]] = math.exp(float(bigram[2]))
        self.words = list(self.bigram_dict.keys())
        self.generate_sentence(None,0)
        print(' '.join(self.sentence))
        
    def generate_sentence(self, first_token, iterations): 
        if iterations == self.sentence_length:
            return
 
        if(iterations == 0):
            first_token = self.words[randint(0, len(self.bigram_dict.keys()))]
            self.sentence.append(first_token)
            return self.generate_sentence(first_token, 1)
            
        bigrams_for_word = list(self.bigram_dict[first_token].items())
         
        probs = []
        for tuple in bigrams_for_word:
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
                next_token = bigrams_for_word[index][0]
                break
        self.sentence.append(next_token)
        iterations += 1
        return self.generate_sentence(next_token, iterations)
    


