import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
    
    
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(root='../data', train=True, transform=train_transform, download=True)
test_dataset = MNIST(root='../data', train=False, transform=test_transform, download=True)

# Insert backdoor trigger: a white square at the corner
def insert_trigger(x):
    x[0, 0:5, 0:5] = 1
    return x

poisoned_indices = np.random.choice(len(train_dataset), size=1000, replace=False)
for i in poisoned_indices:
    train_dataset.data[i] = insert_trigger(train_dataset.data[i])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def invert_trigger(model, initial_trigger, num_iterations=1000, learning_rate=0.1):
    trigger = initial_trigger.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([trigger], lr=learning_rate)
    
    for iteration in range(num_iterations):
        model.zero_grad()
        optimizer.zero_grad()
        
        poisoned_data = insert_trigger(initial_trigger.clone())
        poisoned_data = poisoned_data.unsqueeze(0).unsqueeze(0).float()
        output = model(poisoned_data)
        
        loss = -torch.max(output)
        loss.backward()
        
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
    
    return trigger

initial_trigger = torch.zeros((28, 28))
learned_trigger = invert_trigger(model, initial_trigger)

# Visualize the learned trigger
plt.imshow(learned_trigger.detach().numpy(), cmap='gray')
plt.show()

