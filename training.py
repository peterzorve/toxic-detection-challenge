
import torch 
from torch.utils.data import DataLoader, Dataset 
from data_handler import split_data, TrainData, data 
from model import ToxicClassifier 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 


length_of_data = 159571 
max_seq_length = 64 
idx = int(0.7 * length_of_data)


#####################################################################################################################################################
#####################################################################################################################################################

columns_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
data_toxic, data_severe_toxic, data_obscene, data_threat, data_insult, data_identity_hate = split_data(data)  

train_data_toxic = data_identity_hate.iloc[ : 2000 ].reset_index(drop=True)
test_data_toxic  = data_identity_hate.iloc[2000 : 3000 ].reset_index(drop=True)

dataset_train = TrainData(train_data_toxic, data_target='identity_hate', max_seq_len=max_seq_length)
dataset_test  = TrainData(test_data_toxic,  data_target='identity_hate', max_seq_len=max_seq_length)


#####################################################################################################################################################
#####################################################################################################################################################


def collation_train(batch, vectorizer=dataset_train.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch]) 
    return inputs, target

def collation_test(batch, vectorizer=dataset_test.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch]) 
    return inputs, target

train_loader = DataLoader(dataset_train, batch_size=32, collate_fn=collation_train)
test_loader  = DataLoader(dataset_test,  batch_size=32, collate_fn=collation_test)

#####################################################################################################################################################
#####################################################################################################################################################



from torch import nn
import torch.optim as optim 

emb_dim = 300

model = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=32)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 30
all_train_losses, all_test_losses, all_accuracies = [],  [], []

for e in range(epochs):
     train_losses, test_losses, running_accuracy = 0, 0, 0

     for i, (sentences_train, labels_train) in enumerate(iter(train_loader)):
          # print(sentences_train.shape)
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
               loss_test = criterion(prediction_test, labels_test) 

               test_losses += loss_test.item()


               prediction_class = torch.argmax(prediction_test, dim=1)
               running_accuracy += accuracy_score(labels_test, prediction_class)
          
          avg_test_loss = test_losses/len(test_loader)
          all_test_losses.append(avg_test_loss)

          avg_running_accuracy = running_accuracy/len(test_loader)
          all_accuracies.append(avg_running_accuracy)


     model.train()


     print(f'Epoch  : {e+1:3}/{epochs}    |   Train Loss:  : {avg_train_loss:.8f}     |  Test Loss:  : {avg_test_loss:.8f}  |  Accuracy  :   {avg_running_accuracy:.4f}')

torch.save({ "model_state": model.state_dict(), 'max_seq_len' : 64, 'emb_dim' : 64, 'hidden1' : 32, 'hidden2' : 32}, 'trained_model_IDENTITY_HATE')

plt.plot(all_train_losses, label='Train Loss')
plt.plot(all_test_losses,  label='Test Loss')
plt.plot(all_accuracies,   label='Accuracy')

plt.legend()
plt.show()
