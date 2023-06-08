import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F1
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm

class customDataset(Dataset):
    def __init__(self, videoList, classInd, subsample, size=(240, 320)):
        with open(videoList) as f:
            self.videoList = f.read().splitlines()
        with open(classInd) as f:
            classList = f.read().splitlines()
            self.encodeClass = {x.split(" ")[1] : int(x.split(" ")[0]) - 1 for x in classList}
            self.decodeClass = {int(x.split(" ")[0]) -1 : x.split(" ")[1] for x in classList}
        self.subsample = subsample
        self.n = len(classList)
        self.size = size

    def __len__(self):
        return len(self.videoList)

    def __getitem__(self, idx):
        videoPath = self.videoList[idx].split(" ")[0]
        label = self.encodeClass[videoPath.split("/")[0]]
        video, _, _ = torchvision.io.read_video("./UCF-101/"+videoPath, pts_unit='sec', output_format="TCHW")
        video = video[np.linspace(0, len(video)-1, self.subsample, dtype="int")]
        if video.shape[2:] != self.size:
            video = F1.resize(video, size=self.size, antialias=False)
        video = F1.rgb_to_grayscale(video).transpose(0,1)
        return video/255, label

trainingData = customDataset("trainlist01.txt", "classInd.txt", 8)
testData = customDataset("testlist01.txt", "classInd.txt", 8)

trainDataloader = DataLoader(trainingData, batch_size=32, shuffle=True, pin_memory=True)
validationDataloader = DataLoader(testData, batch_size=32, shuffle=False, pin_memory=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3)
        self.conv2 = nn.Conv3d(16, 32, 3)
        self.conv3 = nn.Conv3d(32, 64, 3)
        self.conv4 = nn.Conv3d(64, 128, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.maxpool = nn.MaxPool3d((1, 3, 3))
        self.fc1 = nn.Linear(11264, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, trainingData.n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def accuracy(outputs, labels):
    with torch.no_grad():
        eq = outputs.argmax(1).to("cpu") == labels
        return (eq.sum() / eq.numel()).numpy()

net = Net().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

lossTrainHist = []
lossValidationHist = []
accuracyTrainHist = []
accuracyValidationHist = []
minLoss = float('inf')

for epoch in range(300):
    print("epoch",epoch+1)
    
    runningTrainLoss = 0.0
    runningValidationLoss = 0.0
    
    net.train()
    print("train")
    for i, data in enumerate(tqdm(trainDataloader), 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs.to("cuda"))
        
        lossTrain = criterion(outputs, labels.to("cuda"))
        lossTrain.backward()
        optimizer.step()
        
        runningTrainLoss += lossTrain.item()
        accuracyTrainHist.append(accuracy(outputs, labels))
        
    lossTrainHist.append(runningTrainLoss/(i+1))
    print(lossTrainHist[-1], accuracyTrainHist[-1])

    net.eval()
    print("eval")
    for i, data in enumerate(tqdm(validationDataloader), 0):
        inputs, labels = data
        with torch.no_grad():
            outputs = net(inputs.to("cuda"))

        lossValidation = criterion(outputs, labels.to("cuda"))
        
        runningValidationLoss += lossValidation.item()
        accuracyValidationHist.append(accuracy(outputs, labels))
        
    lossValidationHist.append(runningValidationLoss/(i+1))
    print(lossValidationHist[-1], accuracyValidationHist[-1])
    
    if minLoss>lossValidationHist[epoch]:
        minLoss = lossValidationHist[epoch]
        bestWeights1 = net.state_dict().copy()
        epochSave = epoch