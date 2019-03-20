from konlpy.tag import Okt 
from pprint import pprint
from collections import namedtuple
from sklearn.linear_model import LogisticRegression
import gensim
import pickle
import nltk

pos_tagger = Okt()

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # remove header
    return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

# Data preprocessing. Execute only once
def import_data():
    train_data = read_data('ratings_train.txt')
    test_data = read_data('ratings_test.txt')

    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

    save_obj(train_data, 'train_docs.obj')
    save_obj(test_data, 'test_docs.obj')

def save_obj(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def read_obj(filename):
    return pickle.load(open(filename, 'rb'))

TaggedDocument = namedtuple('TaggedDocument', 'words tags')

train_docs = read_obj('train_docs.obj')
test_docs = read_obj('test_docs.obj')

tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

model = gensim.models.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, epochs=5, seed=1234)
model.build_vocab(tagged_train_docs)

# Train documnet vectors
model.train(tagged_train_docs, epochs=model.epochs, total_examples=model.corpus_count)

pprint(model['공포/Noun']) # raw NumPy vector of a word
pprint(model.wv.most_similar('공포/Noun'))
pprint(model.wv.most_similar('ㅋㅋ/KoreanParticle'))
pprint(model.wv.most_similar(positive=['여자/Noun', '왕/Noun'], negative=['남자/Noun']))

train_x = [model.infer_vector(doc.words) for doc in tagged_train_docs][:10000]
train_y = [doc.tags[0] for doc in tagged_train_docs][:10000]
test_x = [model.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]

# sentiment classification
print("LogisticRegression")
classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y)
print(classifier.score(test_x, test_y))
