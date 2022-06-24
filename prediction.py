
import torch 
from model import ToxicClassifier
# from training import collation_test 
from torchtext.vocab import FastText 
from torch import optim 
import torch.nn as nn 
from data_handler import front_padding, encoder, preprocessing # , fasttext, DataLoader
import numpy as np 


trained_model_TOXIC         = torch.load('trained_model_TOXIC')
trained_model_SEVERE_TOXIC  = torch.load('trained_model_SEVERE_TOXIC')
trained_model_OBSCENE       = torch.load('trained_model_OBSCENE')
trained_model_THREAT        = torch.load('trained_model_THREAT')
trained_model_INSULT        = torch.load('trained_model_INSULT')
trained_model_IDENTITY_HATE = torch.load('trained_model_IDENTITY_HATE')


model_state_TOXIC         = trained_model_TOXIC['model_state']
model_state_SEVERE_TOXIC  = trained_model_SEVERE_TOXIC['model_state']
model_state_OBSCENE       = trained_model_OBSCENE['model_state']
model_state_THREAT        = trained_model_THREAT['model_state']
model_state_INSULT        = trained_model_INSULT['model_state']
model_state_IDENTITY_HATE = trained_model_IDENTITY_HATE['model_state']


max_seq_length, emb_dim = 64, 300

model_TOXIC         = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)
model_SEVERE_TOXIC  = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)
model_OBSCENE       = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)
model_THREAT        = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)
model_INSULT        = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)
model_IDENTITY_HATE = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)


model_TOXIC.load_state_dict(model_state_TOXIC) 
model_SEVERE_TOXIC.load_state_dict(model_state_SEVERE_TOXIC)
model_OBSCENE.load_state_dict(model_state_OBSCENE)
model_THREAT.load_state_dict(model_state_THREAT)
model_INSULT.load_state_dict(model_state_INSULT)
model_IDENTITY_HATE.load_state_dict(model_state_IDENTITY_HATE)



fasttext = FastText("simple") 

comment = 'more i cant make any real suggestions on improvement - i wondered if the section statistics should be later on, or a subsection of types of accidents  -i think the references may need tidying so that they are all in the exact same format'


features = front_padding(encoder(preprocessing(comment), fasttext), max_seq_length) 


# print(type(features))
# print(features)


features = np.array(features)
# print(type(features))
# print(features)


features = torch.from_numpy(features)
# print(type(features))
# print(features)


model_TOXIC.eval()
with torch.no_grad():
     prediction = model_TOXIC.forward(features)
     print(prediction)




# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.003)






# def get_prediction(image, model, classes_dict):
#      model.eval()
#      with torch.no_grad():
#           probs = torch.exp(model(image))
#           prob, pred = torch.max(probs, dim=1)
#           print(f'This image is a {classes_dict[pred.item()]}  [{prob.item() * 100}]')








