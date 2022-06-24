

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import re 
import spacy 
from torchtext.vocab import FastText
from collections import Counter
import torch 
from torch.utils.data import DataLoader, Dataset 
import torch.optim as optim 
import matplotlib.pyplot as plt 



"""
class ToxicClassifier(nn.Module):
     def __init__(self, max_seq_len=32, emb_dim=300, hidden=64):
          super(ToxicClassifier, self).__init__()
          self.input_layer   = nn.Linear(max_seq_len*emb_dim, hidden)
          self.first_hidden  = nn.Linear(hidden, hidden)
          self.second_hidden = nn.Linear(hidden, hidden)
          self.third_hidden  = nn.Linear(hidden, hidden)
          
          
          self.output_1 = nn.Linear(hidden, 1)
          self.output_2 = nn.Linear(hidden, 1)
          self.output_3 = nn.Linear(hidden, 1)
          self.output_4 = nn.Linear(hidden, 1)
          self.output_5 = nn.Linear(hidden, 1)
          self.output_6 = nn.Linear(hidden, 1)


     def forward(self, inputs):
          x = F.relu(self.input_layer(inputs.squeeze(1).float()))

          x = F.relu(self.first_hidden(x))
          x = F.relu(self.first_hidden(x))
          x = F.relu(self.second_hidden(x))
          x = F.relu(self.third_hidden(x))

 
          output_1 = torch.sigmoid(self.output_1(x))
          output_2 = torch.sigmoid(self.output_2(x))
          output_3 = torch.sigmoid(self.output_3(x))
          output_4 = torch.sigmoid(self.output_4(x))
          output_5 = torch.sigmoid(self.output_5(x))
          output_6 = torch.sigmoid(self.output_6(x))

          return np.array([output_1, output_2, output_3, output_4, output_5, output_6])

"""


data = pd.read_csv(r'C:\Users\ritth\code\Strive\toxic-detection-challenge\train.csv\train.csv')
data = data.drop('id', axis=1)

#################################################################################################################
#################################################################################################################
# data = data.iloc[ : 3000 ].reset_index(drop=True)
#################################################################################################################
#################################################################################################################


def data_engineering(data):
     column_name, no, yes = [], [], []
     for i in data.columns:
          if i == 'comment_text':
               continue
          else:
               column_name.append(i),   no.append(data[i].value_counts()[0]),   yes.append(data[i].value_counts()[1])
     return column_name, no, yes


def preprocessing(data):
     data['comment_text'] = data['comment_text'].apply(lambda text: text.lower()) 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.strip()) 
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\n', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\'', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('"', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('/', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('.', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('-', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('=', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\w*\d\w*\*', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub(r'[^\x00-\x7f]', r' ', text)) 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.strip())
     return data 

data = (preprocessing(data))


def shuffle_data(dataset):
     return dataset.sample(frac=1).reset_index(drop=True)

data = shuffle_data(data)




nlp = spacy.load("en_core_web_sm")

def preprocessing(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return tokens



def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def front_padding(list_of_indexes, max_seq_len, padding_index=0):
    new_out = (max_seq_len - len(list_of_indexes))*[padding_index] + list_of_indexes
    return new_out[:max_seq_len]  


fasttext = FastText("simple")


max_seq_length = 50

class TrainData(Dataset):
     def __init__(self, data, max_seq_len=max_seq_length): # data is the input data, max_seq_len is the max lenght allowed to a sentence before cutting or padding
          self.max_seq_len = max_seq_len
          
          counter = Counter()
          train_iter = iter(data['comment_text'].values)
          self.vec = FastText("simple")

          self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
          self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
          self.vectorizer = lambda x: self.vec.vectors[x]

          
          features = [front_padding(encoder(preprocessing(sequence), self.vec), max_seq_len) for sequence in data['comment_text'].tolist()]
          self.features = features

          # print(type(self.features))

          # self.target = data[data_target]
          self.target = list(data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values)
          # print(type(self.target))
     
     def __len__(self):
          return len(self.features)
     
     def __getitem__(self, i):
          assert len(self.features[i]) == self.max_seq_len
          return self.features[i], self.target[i]


length_of_data = 159571 
max_seq_length = 64 
idx = int(0.7 * length_of_data)

data_train = data.iloc[      : 50000].reset_index(drop=True)
data_test  = data.iloc[50000 : 80000].reset_index(drop=True)

dataset_train = TrainData(data_train)
dataset_test  = TrainData(data_test)

def collation_train(batch, vectorizer=dataset_train.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.tensor([item[1] for item in batch]).float()
    return inputs, target

def collation_test(batch, vectorizer=dataset_test.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.tensor([item[1] for item in batch]).float()
    return inputs, target

train_loader = DataLoader(dataset_train, batch_size=16, collate_fn=collation_train)
test_loader  = DataLoader(dataset_test,  batch_size=16, collate_fn=collation_test)


class ToxicClassifier(nn.Module):
     def __init__(self, max_seq_len=32, emb_dim=300, hidden=64):
          super(ToxicClassifier, self).__init__()
          self.input_layer   = nn.Linear(max_seq_len*emb_dim, hidden)
          self.first_hidden  = nn.Linear(hidden, hidden)
          self.second_hidden = nn.Linear(hidden, hidden)
          self.third_hidden  = nn.Linear(hidden, hidden)

          self.output = nn.Linear(hidden, 6)
          
          self.sigmoid    = nn.Sigmoid()


     def forward(self, inputs):
          x = F.relu(self.input_layer(inputs.squeeze(1).float()))

          x = F.relu(self.first_hidden(x))
          x = F.relu(self.first_hidden(x))
          x = F.relu(self.second_hidden(x))
          x = F.relu(self.third_hidden(x))

          output = self.output(x)

          return output




emb_dim = 300

model = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=64)
criterion = nn.BCEWithLogitsLoss()


optimizer = optim.Adam(model.parameters(), lr=0.003) 

epochs = 30
all_train_losses, all_test_losses, all_accuracies = [],  [], []

for e in range(epochs):
     train_losses, test_losses, running_accuracy = 0, 0, 0

     for i, (sentences_train, labels_train) in enumerate(iter(train_loader)):
          sentences_train.resize_(sentences_train.size()[0], max_seq_length * emb_dim) 

          optimizer.zero_grad()
          prediction_train = model.forward(sentences_train)   
          loss_train = criterion(prediction_train, labels_train) 
          loss_train.backward()   
          optimizer.step()

          train_losses += loss_train.item()

     avg_train_loss = train_losses/len(train_loader)
     all_train_losses.append(avg_train_loss)


     model.eval()
     with torch.no_grad():
          for i, (sentences_test, labels_test) in enumerate(iter(test_loader)):
               sentences_test.resize_(sentences_test.size()[0], max_seq_length * emb_dim)

               prediction_test = model.forward(sentences_test) 
               probability_test = torch.sigmoid(prediction_test)

               loss_test = criterion(prediction_test, labels_test) 

               test_losses += loss_test.item()

               classes = probability_test > 0.5

               running_accuracy += ((classes == labels_test).all(dim=1)).sum()/len(labels_test)

          avg_test_loss = test_losses/len(test_loader)
          all_test_losses.append(avg_test_loss)

          avg_running_accuracy = running_accuracy/len(test_loader)
          all_accuracies.append(avg_running_accuracy)


     model.train()


     print(f'Epoch  : {e+1:3}/{epochs}    |   Train Loss:  : {avg_train_loss:.8f}     |  Test Loss:  : {avg_test_loss:.8f}  |  Accuracy  :   {avg_running_accuracy:.4f}')

torch.save({ "model_state": model.state_dict(), 'max_seq_len' : 64, 'emb_dim' : 64, 'hidden1' : 32, 'hidden2' : 32}, 'TRAINED_MODEL')

plt.plot(all_train_losses, label='Train Loss')
plt.plot(all_test_losses,  label='Test Loss')
plt.plot(all_accuracies,   label='Accuracy')
plt.legend()
plt.show()





