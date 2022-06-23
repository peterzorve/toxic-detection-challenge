

from torch import nn
import torch.nn.functional as F
emb_dim = 300

class ToxicClassifier(nn.Module):
    def __init__(self, max_seq_len, emb_dim, hidden=32):
        super(ToxicClassifier, self).__init__()
        self.input_layer   = nn.Linear(max_seq_len*emb_dim, hidden)
        self.first_hidden  = nn.Linear(hidden, hidden)
        self.second_hidden = nn.Linear(hidden, hidden)
        self.third_hidden  = nn.Linear(hidden, 2)
        self.output        = nn.Softmax(dim=1)
    
    def forward(self, inputs):
        x = F.relu(self.input_layer(inputs.squeeze(1).float()))
        x = F.relu(self.first_hidden(x))
        x = F.relu(self.second_hidden(x))
        x = self.third_hidden(x)

        return self.output(x)