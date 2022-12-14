import os
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from transformers import ViTFeatureExtractor
import requests
from PIL import Image
import glob
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, hamming_loss
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pathlib
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import seaborn as sns
import os
from sklearn.manifold import TSNE
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

num_epochs=5

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

outDir = "/home/christos/results/"


class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        #self.nf = 768

        self.fc1 = nn.Sequential(
            nn.Linear(768, 384),
            #nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )

   

        #self.mha = torch.nn.MultiheadAttention(384, 1, batch_first=True)
	#self.mha = torch.nn.MultiheadAttention(96, 1, batch_first=True)

	
        self.fc = nn.Sequential(
            nn.Linear(384
, 5),
        )


    def forward(self, x):
        x = self.fc1(x)
      #  x = self.fc2(x)

       	#x, _ = self.mha(x,x,x)
        x = self.fc(x)
        return x


class Rosbag(Dataset):
    def __init__(self, path):
        self.imageDim = 224
        
        self.data = []
        dataDF = pd.read_csv(path)
        img = dataDF['filename'].tolist()
        opn = dataDF['Open Path'].tolist()
        obs = dataDF['Obstacle'].tolist()
        stair = dataDF['Staircase'].tolist()
        door = dataDF['Doorway'].tolist()
        human = dataDF['Humans'].tolist()
        row = np.transpose(np.vstack((img, opn, obs, stair, door, human)))
        self.data.extend(row)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataPaths = self.data[idx]
        imgPath = dataPaths[0]
        labels = dataPaths[1:]
        labels = list(map(float, labels))

        transform = transforms.Compose([transforms.Resize(self.imageDim),
            transforms.CenterCrop(self.imageDim),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        img = transform(Image.open(imgPath))

        return img, np.array(labels)

train_loader = DataLoader(Rosbag('/home/christos/code/keras/csvfile_new-Copy1.csv'), batch_size = 64, shuffle = True)
test_loader = DataLoader(Rosbag('/home/christos/code/keras/csvfile_new-Copy1.csv'), batch_size = 64, shuffle = True)


        


model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")

model.eval()

proj = Projector().to(device)

model.classifier = nn.Linear(768, 768)

model.to(device)

i = 0

#comment for unfreeze
for name, param in model.named_parameters():
#     print(i, name)
    if i < 197:
        param.requires_grad = False
    i += 1

allWeights = list(model.parameters()) + list(proj.parameters())
optimizer= SGD(allWeights, lr=0.01, weight_decay=0.0001)


loss_function=nn.BCEWithLogitsLoss() 

#Model training and saving best model


best_accuracy = 0.0
thresholds = [.2, .5, .5, .75, .60]
for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        #print(i)
#         images.to(device), labels.to(device)
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        # print(outputs.size())
        outputs = model(images).logits
        
        outputs = proj(outputs)
        
        loss = loss_function(outputs, labels)
        #
        loss.backward()
        optimizer.step()
      
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs, 1)



#     train_accuracy = train_accuracy / train_count
    train_accuracy = 0
    train_loss = train_loss / len(train_loader)

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    labels_list = []
    prediction_list = []
    x_list = []
    y_list = []

    for i, (images, labels) in enumerate(test_loader):
        
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            labels_list.extend(np.array(labels.cpu()))

        outputs = model(images).logits
        x_list.append(outputs)
        y_list.append(labels)
        outputs = proj(outputs)

        prediction = np.array(outputs.cpu().detach())
        for i in range(len(prediction)):
            for j in range(len(prediction[0])):
                prediction[i][j] = 1 if prediction[i][j] > thresholds[j] else 0
            
        prediction_list.extend(prediction)
        
        
           
        test_accuracy += hamming_loss(prediction, labels.cpu().detach())

    test_accuracy = test_accuracy / len(test_loader)


    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Hamming score: ' + str(test_accuracy))

    
# print(labels_list)
labels_list = np.array(labels_list)
prediction_list = np.array(prediction_list)


clr = classification_report(labels_list, prediction_list)
print(clr)

# mcm = multilabel_confusion_matrix(labels_list,prediction_list)


# print(mcm)



        
