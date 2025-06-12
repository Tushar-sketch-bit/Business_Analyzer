import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rag_ai.data_load_agent import DataLoadAgent
import pandas as pd
class AutoFeatureGenerator(nn.Module):
    """auto feature generator class"""
    def __init__(self, num_features):
        super(AutoFeatureGenerator, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)  # input layer (num_features) -> hidden layer (128)
        self.fc2 = nn.Linear(128, 128)  # hidden layer (128) -> hidden layer (128)
        self.fc3 = nn.Linear(128, num_features)  # hidden layer (128) -> output layer (num_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AutoFeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Initialize the model and dataset
DataLoadAgent(file_path='C:\Users\pagim\Downloads\Analyzer\Main_Project\Business_Analyzer\data\sales_data_sample.csv')
df=DataLoadAgent.get_dataframe()

num_features = df.shape[1]
model = AutoFeatureGenerator(num_features=10)  # Replace with the actual number of features
tensor=torch.from_numpy(df.values).float()


dataset = AutoFeatureDataset(tensor)

# Define the data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # loop over the dataset multiple times
    for i, batch in enumerate(data_loader):
        # forward pass
        outputs = model(batch)
        loss = criterion(outputs, batch)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss at each 100 mini-batches
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, 10, i+1, len(data_loader), loss.item()))

# Use the trained model to generate automatic features
auto_features = model(tensor)