from __future__ import division
import nltk
import operator
import math
import argparse
from random import randint, uniform
from collections import defaultdict

class Bigram:
    def __init__(self, tokens = None):
        if tokens:
            self.tokens = tokens
            self.total_words = len(tokens)
            self.unigram_count = defaultdict(int)
            self.uniq_words = set()
            self.bigrams = nltk.ngrams(tokens,2)
        else:
            self.bigram_dict = defaultdict(lambda: defaultdict(int))
            self.words = [] 
            self.sentence = []
            self.sentence_length = 20

    def generate_model(self):
        total_tokens = len(self.tokens)
        
        bigram_count = defaultdict(int)

        for token in self.tokens:
            self.uniq_words.add(token)
            self.unigram_count[token.strip()] += 1

        for bigram in self.bigrams:
            bigram_count[bigram] += 1
        
        rows_to_print = []
        
        # rows_to_print.append(str(len(self.uniq_words)) + " " + str(self.total_words))
        
        # for index, unigram in enumerate(self.unigram_count):
            # rows_to_print.append(str(index) + " " + str(unigram) + " " + str(self.unigram_count[unigram]))
        # print(rows_to_print)
        for bigram in bigram_count:
            tmp = ""
            for entry in bigram:
                tmp += entry + " "
            log_prob = "{0:.15f}".format(math.log(bigram_count[bigram] / self.unigram_count[bigram[0]]))

            rows_to_print.append(tmp.strip()  + " " + str(log_prob))
        
        return rows_to_print
        
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
            print(len(self.bigram_dict.keys()))
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
    


