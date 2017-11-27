from api import Api
import nltk
import codecs
import argparse
from bigram_model import Ngram
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

def read_model_file(filename):
    model = []
    with codecs.open(filename, 'r', 'utf-8') as text_file:
        model = str(text_file.read()).split('\n')
    return model 
 

def write_model_to_file(filename, model):    
    with codecs.open(filename, 'w', 'utf-8' ) as f:
        for row in model: f.write(row + '\n')

def main():
    # api = Api()

    parser = argparse.ArgumentParser(description='Input')
    
    parser.add_argument('--create_bigram', '-b', type=str,  required=False, help='create bigram model')
    parser.add_argument('--training_file', '-t', type=str,  required=False, help='Training file')
    parser.add_argument('--write_model_file', '-s', type=str,  required=False, help='Where to store the model')

    parser.add_argument('--read_model_file', '-r', type=str,  required=False, help='Read model from file')
    args = parser.parse_args()

    if(args.read_model_file):
        ngram = Ngram(1)
        model = read_model_file(args.read_model_file)
        ngram.read_bigram_model(model)
    elif(args.write_model_file):
        tokens = read_and_tokenize(args.training_file)
        ngram = Ngram(1,tokens) 
        ngram_model = ngram.generate_model()
        write_model_to_file(args.write_model_file, ngram_model)
    else:
        for i in ngram_model: print(i)

if __name__ == '__main__':
    main()
