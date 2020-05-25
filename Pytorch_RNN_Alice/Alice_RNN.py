# Pytorch RNN Alice In wonderland
## LSTM generating text similar to the text of "Alice in Wonderland"

# Imports:

import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm

#%%
if torch.cuda.is_available():
    device = torch.device("cpu") 


#%%
# Defining Dictionary:

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx +=1 
    def __len__(self):
        return len(self.word2idx)
    
#%%
        
class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
    def get_data(self, path, batch_size = 20):
        with open (path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() +['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # we need a 1d tensor containing the indexes of all the words in the file:
        rep_tensor = torch.LongTensor(tokens, device = device)
        index = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() +['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    #print(rep_tensor[index])
                    #print(self.dictionary.word2idx[word])
                    index +=1
        #print(rep_tensor.shape[0])
        # To find out how many batches we need:
        num_batches = rep_tensor.shape[0] // batch_size
        # Remove the remainder:
        rep_tensor = rep_tensor[:num_batches*batch_size]
        #print(rep_tensor.shape[0])

        # return (batch_size, num_batches)
        rep_tensor = rep_tensor.view(batch_size, -1)
        #print(rep_tensor.shape)

        #print("word2idx", len(self.dictionary.word2idx))
        #print(self.dictionary.word2idx)
        #print("idx2word", len(self.dictionary.idx2word))
        #print(self.dictionary.idx2word)
        #print("rep_tensor", rep_tensor.shape)
        #print(rep_tensor)
        
        
        return rep_tensor
#%%

# Defining some parameters:
        
embed_size = 128    # Input featues to the LSTM
hidden_size = 1024  # Number of LSTM units
num_layers = 2
num_epochs = 200
batch_size = 20
timesteps = 30      # We are going to consider 30 previous words.
learning_rate = 0.002
#%%
    
corpus = TextProcess()
rep_tensor = corpus.get_data("alice.txt", batch_size)   # patch and batch_size)
vocab_size = len(corpus.dictionary)
num_of_batches = rep_tensor.shape[1] //timesteps
#%%
print ("rep_tensor shape:", rep_tensor.shape)        
# tensors with indexes of all words (well not all, as we excluded some)
# in each row of rep_tensor there are indices of consecutive 1484 words from the book.
# there are 20 rows, as we will perform 20 batches.

print ("vocab size: ", vocab_size)
     
print("num_of_batches:", num_of_batches)


#%%
class TextGenerator(nn.Module):
    def __init__(self, vocab_size = vocab_size, embed_size = embed_size, hidden_size = hidden_size, num_layers = num_layers):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        # batch_first = True means that inputs and outputs are of shape batch_size*timesteps*features
        # as default it is timesteps*batch_size*features, which is not preferable as other frameworks
        # use the first method
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Do Word EMbedding:
        x = self.embed(x)
        # Reshape the input tensor:
        # x = x.view(batch_size, timesteps, embed_size)
        out, (h,c) = self.lstm(x,h) # h is hidden states
        # Reshape the output from samples, timesteps, output_features to 
        # (batch_size*timesteps, hidden_size) which fits the FC layer
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        # Decode hidden states of all timesteps:
        out = self.linear(out)
        
        return out, (h,c)

#%%
model = TextGenerator() 
if torch.cuda.is_available():
    model = model.cuda()
#%%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

#%%
# training:

for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))

    for i in range (0, rep_tensor.size(1) - timesteps, timesteps):
        # Get mini-batch inputs and targets:
        inputs = rep_tensor[:, i:i+timesteps]
        targets = rep_tensor[:, (i+1):(i+1)+timesteps]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs, _ = model(inputs, states)
        loss = loss_fn(outputs, targets.reshape(-1))
        
        # Backprop, weight update:
        model.zero_grad()
        loss.backward()
        
        #Gradient Cliping: clip_value (float or int) is the maximum  allowed value of the
        # gradient. Gradients are clipped in the range [-clipped_value, clipped_value]
        # This is to prevent the exploding gradient problem.
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        
        step = (i+1)// timesteps
        if step %100 == 0:
            print("Epoch {}/{}, Loss {:.4f}".format(epoch+1, num_epochs, loss.item()))
        
    
#%%
# Testing the model:
with torch.no_grad():
    with open("results.txt", 'w') as f:
         # Set initial hidden and cell states
         state = (torch.zeros(num_layers, 1, hidden_size),
                   torch.zeros(num_layers, 1, hidden_size))
         input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)
         for i in range (500):
             output, _ = model(input, state)
             print(output.shape)
             # Sample a word id from exponential of the output; probability
             prob = output.exp()
             word_id = torch.multinomial(prob, num_samples = 1).item()
             print (word_id)
             # replace the input with sampled word id for the next timestep
             input.fill_(word_id)
             
             #Write the result to the file:
             word = corpus.dictionary.idx2word[word_id]
             word = '\n' if word =='<eos>' else word+ ' '
             f.write(word)
             
             if i %100 ==0:
                 print("Sampled {}/{} words saved to results.txt".format(i+50, 500))
        
        
        
        
        
        
        