import nltk
import math
from collections import defaultdict
from random import randint, uniform

class Unigram:
    def __init__(self, tokens = None):
        if tokens:
            self. tokens = tokens
            self.unigrams = nltk.ngrams(tokens, 1)
            self.unigram_count = defaultdict(int)
            self.get_unigram_count()
        self.unigram_dict = defaultdict(int)
        self.words = []
        self.sentence = []
        self.sentence_length = 20

    def generate_model(self):
        return self.unigram_probs()

    def get_unigram_count(self):
        for token in self.tokens:
            self.unigram_count[token.strip()] += 1

    def unigram_probs(self):
        all_unigrams = len([word_count for word_count in self.unigram_count])
        rows_to_print = []
        for unigram in self.unigram_count:
            log_prob = "{0:.15f}".format(math.log(self.unigram_count[unigram] / all_unigrams))
            rows_to_print.append(unigram + ' ' + str(log_prob))
        return rows_to_print

    def read_unigram_model(self, model):
        for row in model:
            unigram = row.split(' ')
            self.unigram_dict[unigram[0]] = math.exp(float(unigram[1]))
        self.words = list(self.unigram_dict.keys())
        self.generate_sentence(None, 0)
        print(' '.join(self.sentence))

    def generate_sentence(self, first_token, iterations): 
        if iterations == self.sentence_length:
            return
 
        if(iterations == 0):
            first_token = self.words[randint(0, len(self.unigram_dict.keys()))]
            self.sentence.append(first_token)
            return self.generate_sentence(first_token, 1)
            
        prob_sum = sum(list(self.unigram_dict.values()))

        probs = [p/prob_sum for p in self.unigram_dict.values()]
        acc = 0
        prob_array = []
        for p in probs:
            acc += p
            prob_array.append(acc)
        
        keys = list(self.unigram_dict.keys())
        rand_index = uniform(0,1)
        next_token = ''
        for index, value in enumerate(prob_array):
            if rand_index < value:
                next_token = list(self.unigram_dict.keys())[index]
                break
        self.sentence.append(next_token)
        iterations += 1
        return self.generate_sentence(next_token, iterations)
    

