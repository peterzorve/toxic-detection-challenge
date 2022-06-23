import torch
from model import Network
import torch.nn as nn
from torchtext.vocab import FastText
import matplotlib.pyplot as plt







torch.manual_seed(0)

model = Network(32, 300, 56, 32, 12, 1)



# loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.BCELoss()



# train
emb_dim = 300
num_epochs = 10
print_every = 40

train_losses = []
test_losses = []


for epoch in range(num_epochs):
    running_loss = 0
    print(f"Epoch: {epoch+1}/{num_epochs}")

    for i, (sentences, labels) in enumerate(iter(train_loader)):

        sentences.resize_(sentences.size()[0], 32* emb_dim)
        
        model.train()
        optimizer.zero_grad()
        
        output = model.forward(sentences) 
        loss = criterion(output, labels) 
        loss.backward()                  
        optimizer.step()                 
        train_losses.append(loss.detach().item())
        running_loss += loss.item()
        
        if i % print_every == 0:
            print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
            running_loss = 0

        
        model.eval()
        with torch.no_grad():
            running_loss = 0
            
            for i, (test_sentences, test_labels) in enumerate(iter(test_loader)):
                test_sentences.resize_(test_sentences.size()[0], 32* emb_dim)

                test_out = model.forward(test_sentences)
                test_loss = criterion(test_out, test_labels)
                running_loss += test_loss.item()
                test_losses.append(test_loss.detach().item())

        model.train()



plt.plot(train_losses, label= "Train Loss")
plt.plot(test_losses, label= "Test Loss")
plt.xlabel(" Iteration ")
plt.ylabel("Loss value")
plt.legend(loc="upper left")
plt.show()