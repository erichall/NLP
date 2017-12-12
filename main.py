# -*- coding: utf-8 -*-
# coding: utf-8
#from api import Api
import nltk
import codecs
import argparse
from bigram_model import Bigram
from unigram_model import Unigram
from trigram_model import Trigram
from quadragram_model import Quadragram
from lstm_generator import Lstm

#nltk.download('punkt')

def read_and_tokenize(filename):
    tokens = []
    with codecs.open(filename, 'r', 'utf-8') as text_file:
            text = str(text_file.read()).lower().strip()
            tokens = nltk.word_tokenize(text)

    clean_tokens = []
    for token in tokens:
        clean_tokens.append(token.replace("\\", "").replace("\'", "'"))
    return clean_tokens

def read_file(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as text_file:
        return str(text_file.read())

def read_model_file(filename):
    model = []
    with codecs.open(filename, 'r', 'utf-8') as text_file:
        model = str(text_file.read()).split('\n')
    return model 
 

def write_model_to_file(filename, model):    
    with codecs.open(filename, 'w', 'utf-8' ) as f:
        for row in model: f.write(row + '\n')

def main():
    #api = Api()
    #exit(0)

    parser = argparse.ArgumentParser(description='Input')
   
    # training file is needed for all inputs except when reading a premade model
    parser.add_argument('--training_file', '-t', type=str,  required=False, help='Training file')


    parser.add_argument('--unigram', '-uni', action='store_true',  required=False, help='create unigram model')
    parser.add_argument('--bigram', '-big', action='store_true',  required=False, help='create bigram model')

    parser.add_argument('--trigram', '-tri', action='store_true',  required=False, help='create bigram model')

    parser.add_argument('--quadragram', '-quad', action='store_true',  required=False, help='create bigram model')

    parser.add_argument('--lstm', '-lstm', action='store_true',  required=False, help='RNN generator')

    parser.add_argument('--write_model_file', '-s', type=str,  required=False, help='Where to store the model')

    parser.add_argument('--ephocs', '-e', type=str,  required=False, help='number of ephocs')
    parser.add_argument('--read_model_file', '-r', type=str,  required=False, help='Read model from file')
    args = parser.parse_args()
    
    tokens = []
    if(args.training_file):
        tokens = read_and_tokenize(args.training_file)

    if args.unigram:
        unigram = Unigram(tokens)
        unigram_model = unigram.generate_model()
        write_model_to_file(args.write_model_file, unigram_model)
        unigram.read_unigram_model(unigram_model)
    elif args.bigram:
        bigram = Bigram(tokens)
        bigram_model = bigram.generate_model()
        write_model_to_file(args.write_model_file, bigram_model)
        bigram.read_bigram_model(bigram_model)
    elif args.trigram:
        trigram = Trigram(tokens)
        trigram_model = trigram.generate_model()
        write_model_to_file(args.write_model_file, trigram_model)
        trigram.read_trigram_model(trigram_model)
    elif args.quadragram:
        quadragram = Quadragram(tokens)
        quadragram_model = quadragram.generate_model()
        write_model_to_file(args.write_model_file, quadragram_model)
        quadragram.read_quadragram_model(quadragram_model)
    elif args.lstm:
        lstm = Lstm(read_file(args.training_file), args.write_model_file, args.ephocs)

    # if(args.read_model_file):
    #     ngram = Ngram(2)
    #     model = read_model_file(args.read_model_file)
    #     ngram.read_bigram_model(model)
    # elif(args.write_model_file):
    #     ngram = Ngram(2,tokens) 
    #     ngram_model = ngram.generate_model()
    #     print(ngram_model)
    #     write_model_to_file(args.write_model_file, ngram_model)
    # else:
    #     for i in ngram_model: print(i)

if __name__ == '__main__':
    main()
