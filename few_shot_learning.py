import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFewShotLearner(nn.Module):
    def __init__(self, embedding_size=512):
        super(EnhancedFewShotLearner, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, embedding_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class EnhancedFewShotTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.TripletMarginLoss(margin=0.2)

    def train(self, anchor, positive, negative):
        self.model.train()
        self.optimizer.zero_grad()
        
        anchor_out = self.model(anchor)
        positive_out = self.model(positive)
        negative_out = self.model(negative)
        
        loss = self.criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, anchor, positive, negative):
        self.model.eval()
        with torch.no_grad():
            anchor_out = self.model(anchor)
            positive_out = self.model(positive)
            negative_out = self.model(negative)
            
            loss = self.criterion(anchor_out, positive_out, negative_out)
        
        return loss.item()