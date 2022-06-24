import torch.nn as nn
import torch.nn.functional as F





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