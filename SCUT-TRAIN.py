import torch
import time
import os
from torch.autograd import Variable
import numpy as np
from model import TwoStreamTransformer
import torch.nn as nn
import time
from torch.utils.data import Dataset,DataLoader
def savedata(root):
    image, image1,lebal = [], [],[]
    vecs_pt = torch.load(root)
    image,image1,lebal=vecs_pt[0],vecs_pt[1],vecs_pt[2]
    c=[image, image1, lebal]
    return c
class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = dataset[0]
        self.image = dataset[1]
        #self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.label = dataset[2]

    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx]), self.label[idx]
traindir = "Couch_GB2/trnCouch_GB2.pth"
testdir = "Couch_GB2/tstCouch_GB2.pth"
time0=time.time()
train=savedata(traindir)
test=savedata(testdir)
train_dataset = Rumor_Data(train)
test_dataset = Rumor_Data(test)
lr = 0.002
seq_length = int(1120)
certion = nn.CrossEntropyLoss()
batch_size = 256
train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
if torch.cuda.is_available():
    print('use gpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoStreamTransformer(3008)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
from torch.optim.lr_scheduler import MultiStepLR
lr_scheduler = MultiStepLR(optimizer,
                               milestones=[15,30],
                               gamma=0.1)
certion = nn.CrossEntropyLoss()
lrlist=[]
def train(ep):
    global steps
    model.train()
    train_loss = 0
    now = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_txt1,train_txt2,target=Variable(data[0].to(device)),Variable(data[1].to(device)),Variable(target.to(device))
        optimizer.zero_grad()
        train_txt1 = train_txt1.view(-1, 1, seq_length)
        train_txt2 = torch.unsqueeze(train_txt2, dim=1)
        output = model(train_txt1,train_txt2)
        output = torch.squeeze(output, dim=0)
        loss = certion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss
    traintime = time.time()
    print(traintime - now)
    lr_scheduler.step(ep)
def test(ep):
    model.eval()
    test_loss = 0
    correct = 0
    if ep ==1:
        torch.save(model.state_dict(),'papernet5.mdl')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            starttime = time.time()
            test_txt1,test_txt2,target=Variable(data[0].to(device)),Variable(data[1].to(device)),Variable(target.to(device))
            test_txt1 = test_txt1.view(-1, 1, seq_length)
            test_txt2=torch.unsqueeze(test_txt2,dim=1)
            output = model(test_txt1,test_txt2).to(device)
            endtime = time.time()
            #print(f"It took {endtime-starttime:.3f} seconds to compute")
            test_loss += certion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        acc=float(correct / len(test_loader.dataset))
        return test_loss
epochs =45
if __name__ == '__main__':
    for epoch in range(1, epochs):
        print(epoch)
        time3=time.time()
        train(epoch)
        test(epoch)
        time4=time.time()
        print('训练时间',time4-time3)
