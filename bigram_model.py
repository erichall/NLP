import nltk
import operator
import math
import argparse
from collections import defaultdict

class Bigram:
    def __init__(self, tokens):
        self.tokens = tokens
        self.total_words = len(tokens)
        self.unigram_count = defaultdict(int)
        self.uniq_words = set()
        self.bigrams = nltk.ngrams(tokens,2) 

    def probabilities(self):
        total_tokens = len(self.tokens)
        
        bigram_count = defaultdict(int)

        for token in self.tokens:
            self.uniq_words.add(token)
            self.unigram_count[token.strip()] += 1

        for bigram in self.bigrams:
            bigram_count[bigram] += 1
        
        rows_to_print = []
        
        rows_to_print.append(str(len(self.uniq_words)) + " " + str(self.total_words))
        
        for index, unigram in enumerate(self.unigram_count):
            rows_to_print.append(str(index) + " " + str(unigram) + " " + str(self.unigram_count[unigram]))
        # print(rows_to_print)
        for bigram in bigram_count:
            tmp = ""
            for entry in bigram:
                tmp += entry + " "
            log_prob = "{0:.15f}".format(math.log(bigram_count[bigram] / self.unigram_count[bigram[0]]))

            rows_to_print.append(tmp + " " + str(bigram_count[bigram]) + " " + str(log_prob))
        
        return rows_to_print
        
        

        
