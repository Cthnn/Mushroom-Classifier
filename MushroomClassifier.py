import glob
import numpy as np
import cv2
import torch
import gc
import random
import sys
from os import path
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #OUR CUSTOM MODEL REMOVING UNECESSARY CONVOLUTIONS
        self.conv1 = nn.Conv2d(3,16,2)
        self.conv2 = nn.Conv2d(16,32,2)
        self.conv3 = nn.Conv2d(32,64,2)
        self.conv4 = nn.Conv2d(64,128,2)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(0.4)
        self.dense1 = nn.Linear(13*13*128,4096)
        self.dense2 = nn.Linear(4096,5)
        

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1,13*13*128)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.softmax(self.dense2(x))
        return x

def img_norm(img):
    if(np.std(img) == 0):
        print("test")
        sys.exit()
    else:
        res = ((img-np.mean(img))/np.std(img))
        return res

def load_dataset(path, img_size,batch_num=1, shuffle=False, augment=False, is_color=False,
                rotate_90=False, zero_centered=False):
    
    data = []
    labels = []
    if is_color:
        channel_num = 3
    else:
        channel_num = 1
    
    for id, class_name in class_names.items():
        img_path_class = glob.glob(path + class_name + '/*.jpg')
        augment_num = 1
        if(augment):
            augment_num = 5
        labels.extend([id]*len(img_path_class)*augment_num)
        for filename in img_path_class:
            if is_color:
                img = cv2.imread(filename)
            else:
                img = cv2.imread(filename, 0)
            
            img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
            data.append(img_norm(img))
            if (augment):
                data.append(img_norm(cv2.flip(img,1)))
                data.append(img_norm(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)))
                data.append(img_norm(cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)))
                data.append(img_norm(cv2.rotate(img,cv2.ROTATE_180)))
    if zero_centered:
      data = data - np.mean(data)

    if shuffle:
        bundle = list(zip(data, labels))
        random.shuffle(bundle)
        data, labels = zip(*bundle)
    
    if batch_num > 1:
        batch_data = []
        batch_labels = []

        
        for i in range(int(len(data) / batch_num)):
            minibatch_d = data[i*batch_num: (i+1)*batch_num]
            minibatch_d = np.reshape(minibatch_d, (batch_num, channel_num, img_size[0], img_size[1]))
            batch_data.append(torch.from_numpy(minibatch_d))

            minibatch_l = labels[i*batch_num: (i+1)*batch_num]
            batch_labels.append(torch.LongTensor(minibatch_l))
        data, labels = batch_data, batch_labels 
    
    return zip(batch_data, batch_labels)

class_names = [name[13:] for name in glob.glob('./data/train/*')]
class_names = dict(zip(range(len(class_names)), class_names))

img_size = (224, 224)
batch_num = 50 
if(not path.exists('./ShroomModel.pkl')):
    trainloader = list(load_dataset('./data/train/', img_size, batch_num=batch_num, shuffle=True, 
                                        augment=True,is_color=True, zero_centered=True))
    train_num = len(trainloader)
    print("%d batches" % train_num)
else:
    testloader = list(load_dataset('./data/test/', img_size, batch_num=batch_num,is_color=True))
    test_num = len(testloader)
    print("%d batches" % test_num)

if(not path.exists('./ShroomModel.pkl')):
    model = CNN().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(120):
        for i, data in enumerate(trainloader, 0):
            inputs,labels = data
            optimizer.zero_grad()
            out = model((inputs.float()).cuda())
            loss = loss_func(out.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()
    torch.save(model, './ShroomModel.pkl')
    print("Model Saved")
else:
    model = torch.load('./ShroomModel.pkl')
    print("Model Loaded")
    test_labels = []
    res = []
    for i, data in enumerate(testloader,0):
        inputs, labels = data
        outputs = model((inputs.float()).cuda())
        _,pred = torch.max(outputs, 1)
        test_labels = test_labels + labels.tolist()
        res = res + pred.tolist()
    print("Custom Model: ",accuracy_score(test_labels,res)*100)