import torch
import torch.nn as nn

class bp_network_basic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(bp_network_basic, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(hidden_dim, num_class)
        )
    
    def forward(self, x):
        output = self.classifier(x)
        return output

