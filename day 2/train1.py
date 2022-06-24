import torch 
from model import ToxicClassifier
from data_handler1 import train_loader, test_loader
import torch.nn as nn
import matplotlib.pyplot as plt 






# model
emb_dim = 300
max_seq_length = 60

model = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=64)



# loss
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) 



# train and evaluation
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

               # convert output 0 to 1
               probability_test = torch.sigmoid(prediction_test)
               
               # loss
               loss_test = criterion(prediction_test, labels_test) 
               test_losses += loss_test.item()
               
               # accuracy
               classes = probability_test > 0.5
               running_accuracy += ((classes == labels_test).all(dim=1)).sum()/len(labels_test)

          avg_test_loss = test_losses/len(test_loader)
          all_test_losses.append(avg_test_loss)

          avg_running_accuracy = running_accuracy/len(test_loader)
          all_accuracies.append(avg_running_accuracy)


     model.train()


     print(f'Epoch  : {e+1:3}/{epochs}    |   Train Loss:  : {avg_train_loss:.8f}     |  Test Loss:  : {avg_test_loss:.8f}  |  Accuracy  :   {avg_running_accuracy:.4f}')

     

# save model
torch.save({ "model_state": model.state_dict(), 'max_seq_len' : 64, 'emb_dim' : 64, 'hidden1' : 32, 'hidden2' : 32}, 'TRAINED_MODEL')




plt.plot(all_train_losses, label='Train Loss')
plt.plot(all_test_losses,  label='Test Loss')
plt.plot(all_accuracies,   label='Accuracy')
plt.legend()
plt.show()
