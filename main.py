from api import Api
import nltk
import codecs
import argparse
from bigram_model import Bigram
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

def write_model_to_file(filename, model):    
    with codecs.open(filename, 'w', 'utf-8' ) as f:
        for row in model: f.write(row + '\n')

def main():
    # api = Api()

    parser = argparse.ArgumentParser(description='Input')
    
    parser.add_argument('--create_bigram', '-b', type=str,  required=False, help='create bigram model')
    parser.add_argument('--training_file', '-t', type=str,  required=True, help='Training file (mandatory)')
    parser.add_argument('--store_model_file', '-s', type=str,  required=False, help='Where to store the model (mandatory)')
    args = parser.parse_args()

    tokens = read_and_tokenize(args.training_file)
    bigram = Bigram(tokens) 
    bigram_model = bigram.probabilities()

    if(args.store_model_file):
        write_model_to_file(args.store_model_file, bigram_model)
    else:
        for i in bigram_model: print(i)

if __name__ == '__main__':
    main()
