import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from torch.optim import Adam
import numpy as np
from tqdm.auto import tqdm
import umap
import umap.plot
import pandas as pd
from warnings import filterwarnings

mnist_train = datasets.MNIST(root='mnist/',train=True,transform=ToTensor(),download=True)
mnist_test  = datasets.MNIST(root='mnist/',train=False,transform=ToTensor(),download=True)


train_loader = DataLoader(dataset=mnist_test,batch_size=32)
test_loader = DataLoader(dataset=mnist_train,batch_size=32)

filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise(tensor,std=0.7,mean=0.):
    
    return tensor + torch.normal(mean,std,tensor.size())


_, (example_data, example_targets) = next(enumerate(train_loader))
denoised_example = add_noise(example_data)


def visualize_dataset(example_data,example_targets,channel_size):
    
    if channel_size==3:
        fig = plt.figure()
        
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(np.transpose(example_data[i],(1, 2, 0)))
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        
    if channel_size == 1:
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])


class DenoisingAE(nn.Module):
    
    def __init__(self):
        super(DenoisingAE,self).__init__()
        
        self.encoder = nn.Sequential(
            
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.ReLU(),
        
        )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.ReLU(),
        )
        
    
    def forward(self,x):
        
        x = x.view(x.size(0),-1)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    

model_relu = DenoisingAE()


optimizer = Adam(model_relu.parameters(),lr=1e-3)
criterion = torch.nn.MSELoss()
EPOCH = 20

def train(model,train_dataloader,optimizer,criterion,EPOCH):
    
    with tqdm(total=len(train_dataloader)*EPOCH) as tt:
        
        for epoch in range(EPOCH):

            total_loss, batch_count = 0, 0
            for idx, (batch,_) in enumerate(train_dataloader):

               ##################################
                denoised_data = add_noise(batch)
               ##################################


                output = model(denoised_data)

                loss = criterion(output,batch.view(batch.size(0),-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                batch_count += 1
                tt.update()

            total_loss = total_loss / batch_count
            print(f'{total_loss}')
            
train(model=model_relu,train_dataloader=train_loader,optimizer=optimizer,criterion=criterion,EPOCH=EPOCH)