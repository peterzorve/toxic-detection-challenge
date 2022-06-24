
import pandas as pd 
import re 
import spacy 
from torchtext.vocab import FastText
from collections import Counter
import torch 
from torch.utils.data import DataLoader, Dataset 




# load data
data = pd.read_csv(r'C:\Users\ritth\code\Strive\toxic-detection-challenge\train.csv\train.csv')
data = data.drop('id', axis=1)




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

data = preprocessing(data)




def shuffle_data(dataset):
     return dataset.sample(frac=1).reset_index(drop=True)

data = shuffle_data(data)



# load dictionary
nlp = spacy.load("en_core_web_sm")
fasttext = FastText("simple")




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





class TrainData(Dataset):
     def __init__(self, data, max_seq_len = 60): # data is the input data, max_seq_len is the max lenght allowed to a sentence before cutting or padding
          self.max_seq_len = max_seq_len
          
          counter = Counter()
          train_iter = iter(data['comment_text'].values)
          self.vec = FastText("simple")

          self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
          self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
          self.vectorizer = lambda x: self.vec.vectors[x]

          
          features = [front_padding(encoder(preprocessing(sequence), self.vec), max_seq_len) for sequence in data['comment_text'].tolist()]
          self.features = features
          self.target = list(data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values)
          
     
     def __len__(self):
          return len(self.features)
     
     def __getitem__(self, i):
          assert len(self.features[i]) == self.max_seq_len
          return self.features[i], self.target[i]


# split data
length_of_data = 159571           # hole data
idx = int(0.7 * length_of_data)

data_train = data.iloc[: idx].reset_index(drop=True)
data_test  = data.iloc[idx :].reset_index(drop=True)

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