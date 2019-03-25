import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("The dog ate the banana".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("The dog ate the meat".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("The dog ate the grape".split(), ["DET", "NN", "V", "DET", "NN"]),
]
test_data = [
    ("The dog ate the dog".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("The dog ate the the".split(), ["DET", "NN", "V", "DET", "NN"]),
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 10
HIDDEN_DIM = 6
BATCH_SIZE = 2
SENTENCE_LEN = 5
#input = Variable(torch.randn(batch_size,length,input_size)) # B,T,D  <= batch_first
#output = (batch, length, num_directions * hidden_size)
#hidden = Variable(torch.zeros(1,batch_size,hidden_size)) # 1,B,H    (num_layers * num_directions, batch, hidden_size)
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentences):
        embeds = torch.stack([self.word_embeddings(sentence) for sentence in sentences])
        lstm_out, _ = self.lstm(embeds.view(self.batch_size, SENTENCE_LEN, -1))
        tag_space = self.hidden2tag(lstm_out.view(self.batch_size, SENTENCE_LEN, -1))
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores

def main():
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), BATCH_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for i in range(int(len(training_data) / BATCH_SIZE)):
            sentences = training_data[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
    
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            inputs = torch.stack([prepare_sequence(sentence, word_to_ix) for sentence, _ in sentences])
            targets = torch.stack([prepare_sequence(tags, tag_to_ix) for _, tags in sentences])
    
            # Step 3. Run our forward pass.
            tag_scores = model(inputs)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores[0], targets[0])
            loss.backward(retain_graph=True)
            loss = loss_function(tag_scores[1], targets[1])
            loss.backward()
            optimizer.step()
    
    # See what the scores are after training
    with torch.no_grad():
        sentences = test_data[:2]
        inputs = torch.stack([prepare_sequence(sentence, word_to_ix) for sentence, _ in sentences])
        tag_scores = model(inputs)
    
        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(inputs)
        print(tag_scores)
        print(tag_scores.argmax(dim=-1))
main()
