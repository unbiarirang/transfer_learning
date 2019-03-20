import re
import numpy as np
from konlpy.tag import Okt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pprint
import pickle
import argparse

okt = Okt()
pprint = pprint.PrettyPrinter(indent=2).pprint

# Parse arguments
parser = argparse.ArgumentParser(description='figure training arguments')
parser.add_argument('--embedding-dim', default=10, type=int, required=False)
parser.add_argument('--hidden-dim', default=10, type=int, required=False)
parser.add_argument('-n', '--epoches', default=1, type=int, required=False)
parser.add_argument('--train-size', default=100, type=int, required=False)
parser.add_argument('--test-size', default=100, type=int, required=False)
parser.add_argument('--save-path', default='./checkpoint/model.pt', required=False, help='checkpoint save path')
parser.add_argument('--load-path', default='./checkpoint/model.pt', required=False, help='checkpoint load path')
parser.add_argument('-s', '--is-save', default=False, required=False)
parser.add_argument('-l', '--is-load', default=False, required=False)
args = parser.parse_args()

EMBEDDING_DIM = args.embedding_dim
HIDDEN_DIM = args.hidden_dim
EPOCHES = args.epoches
TRAIN_SIZE = args.train_size
TEST_SIZE = args.test_size
SAVE_PATH = args.save_path
LOAD_PATH = args.load_path
is_save = args.is_save
is_load = args.is_load

TAG_NUM = 2
train_data_path = 'ratings_train.txt'
test_data_path = 'ratings_test.txt'
train_data_obj = 'train_transfer_ko.obj'
test_data_obj = 'test_transfer_ko.obj'

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # remove header
    return data

def read_obj(filename):
    return pickle.load(open(filename, 'rb'))

def save_obj(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

# Data preprocessing. Execute only once
def import_data():
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)

    train_docs = [(row[1], row[2]) for row in train_data]
    test_docs = [(row[1], row[2]) for row in test_data]

    save_obj(train_docs, train_data_obj)
    save_obj(test_docs, test_data_obj)

def preprocessing(sentence, stop_words=[]):
    tokens = [];

    line_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", sentence)
    tokens += okt.morphs(line_text, stem=True)

    if len(stop_words) != 0:
        tokens = [token for token in tokens if token not in stop_words]

    return tokens

def preprocessing_tag(tag):
    return torch.tensor([int(tag)])

def get_word_dict(tokens):
    word_dict = {}
    word_dict['PAD'] = len(word_dict)
    word_dict['UNK'] = len(word_dict)

    for word in tokens:
        if word not in word_dict:
            word_dict[word] = len(word_dict)

    return word_dict

def get_embeddings(word_dict, vec_file, emb_size):
    word_vec = [torch.randn(1, emb_size) for _ in range(len(word_dict))]
    f = open(vec_file, 'r', encoding='utf-8')
    num_vec, max_vec_size = f.readline().split()
    print(num_vec, max_vec_size)

    vec = {}
    for i, line in enumerate(f):
        line = line.split()
        vec[line[0]] = torch.tensor([float(x) for x in line[-emb_size:]]).view(-1, emb_size)
    f.close()

    for word in word_dict:
        lower = word.lower()
        if lower in vec:
            word_vec[word_dict[word]] = vec[lower]

    return word_vec

def word_to_embeds(words, word_dict, word_vec):
    idxs = word_to_idx(words, word_dict)
    return torch.stack([word_vec[idx] for idx in idxs])

def word_to_idx(words, word_dict):
    return [word_dict[word] if word in word_dict else word_dict['UNK'] for word in words]

class LSTMTagger(nn.Module):

    def __init__(self, word_dict, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_dict = word_dict
        self.word_embeddings = get_embeddings(word_dict, 'wiki_ko.vec', embedding_dim)
        self.word_to_embeds = word_to_embeds

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words):
        embeds = self.word_to_embeds(words, self.word_dict, self.word_embeddings)
        lstm_out, _ = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def main():
    train_docs = read_obj('train_transfer_ko.obj')[:TRAIN_SIZE]
    test_docs = read_obj('test_transfer_ko.obj')[:TEST_SIZE]

    stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한']
    tokens_list = [preprocessing(review[0], stop_words) for review in train_docs]
    # flat the token list
    tokens = [item for tokens in tokens_list for item in tokens]
    word_dict = get_word_dict(tokens)
    word_embeddings = get_embeddings(word_dict, 'wiki_ko.vec', EMBEDDING_DIM)

    print('test "word_to_embeds"')
    pprint(word_to_embeds(['더빙', '목소리', '연기'], word_dict, word_embeddings))
    
    if is_load:
        model = torch.load(LOAD_PATH)
    else:
        # new LSTM model
        model = LSTMTagger(word_dict, EMBEDDING_DIM, HIDDEN_DIM, len(word_dict), TAG_NUM)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Train
    for epoch in range(EPOCHES):
        print(epoch, '/', EPOCHES)
        for review, tag in train_docs:
            # Step 1. Remember that Pytorch accumulates gradients.
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is, turn them into
            inputs = preprocessing(review, stop_words)
            if len(inputs) == 0: continue
            targets = preprocessing_tag(tag)

            # Step 3. Run our forward pass.
            tag_scores = model(inputs)

            # Step 4. Compute the loss, gradients, and update the parameters by
            loss = loss_function(tag_scores[-1].view(-1, TAG_NUM), targets)
            loss.backward()
            optimizer.step()
    
    # Save checkpoint
    if is_save:
        torch.save(model, SAVE_PATH)
        print('save completed')

    # Evaluate
    model.eval()
    correct = 0
    total = len(test_docs)
    with torch.no_grad():
        for review, tag in test_docs:
            inputs = preprocessing(review, stop_words)
            if len(inputs) == 0: continue
            tag_scores = model(inputs)
            predicted = tag_scores[-1].argmax(dim=-1)
            if predicted.item() == int(tag):
                correct += 1
    accuracy = (correct / total) * 100
    print(accuracy, '%')

    with open('./log', 'a') as f:
        f.write('embedding-dim: {} hidden-dim: {} epoches: {} accuracy: {}%\n'.format(EMBEDDING_DIM, HIDDEN_DIM, EPOCHES, accuracy))

main()
