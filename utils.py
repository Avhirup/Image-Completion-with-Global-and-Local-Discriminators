import torch.nn as nn
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import random
import numpy as np
from PIL import Image
import os
from torch.autograd import Variable as V
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class TileSent(nn.Module):
    def __init__(self, tiles=32):
        super(TileSent, self).__init__()
        self.tiles = tiles
            
    def forward(self,x,shape=[1,64,64]):
        x=x.view(-1,shape[0],shape[1],shape[2])
        x=torch.cat([x]*self.tiles,dim=1)
        return x

class Concat(nn.Module):
    def forward(self,x,y):
        res=torch.cat((x,y),dim=1)                        
        return res

def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

def one_hot(val,nb_digits=10):
    label=[0]*nb_digits
    label[val]=1
    return np.array(label)

def find_center(m):
    z=np.where(m!=0)
    x,y=z[0],z[1]
    x_min,x_max=np.min(x),np.max(x)
    y_min,y_max=np.min(y),np.max(y)
    x_center=(x_min+x_max)//2
    y_center=(y_min+y_max)//2
    return [x_center,y_center],[x_min,x_max,y_min,y_max]
    

def find_centers(mask):
    try:
        m=mask.numpy()
    except:
        m=mask.cpu().numpy()

    res_c,res_cord=[],[]
    for i in range(m.shape[0]):
        center,coords=find_center(m[i,0])
        res_c.append(center)
        res_cord.append(coords)
    return res_c,res_cord

def find_patches(img_c,mask,patch_size=32,IS_LABEL_ADDED=True):
    center,coords = find_centers(mask)
    resized_images,resized_masks=[],[]
    p=patch_size//2
    for ind,i in enumerate(range(img_c.size()[0])):
        x_center,y_center=center[ind]
        x_min,x_max,y_min,y_max=max(0,abs(x_center-p)),min(64,x_center+p),max(0,abs(y_center-p)),min(64,y_center+p)
        if not IS_LABEL_ADDED:
            img=img_c[ind][:,x_min:x_max,y_min:y_max]
            img=F.upsample_bilinear(img.unsqueeze(0),(patch_size,patch_size))
        else:
            temp=img_c[ind][0,x_min:x_max,y_min:y_max]
            temp=F.upsample_bilinear(temp.unsqueeze(0).unsqueeze(0),(patch_size,patch_size))
            img=torch.cat([temp,img_c[ind][1,0:patch_size,0:patch_size].unsqueeze(0).unsqueeze(0)],dim=1)
        m=mask[ind][:,x_min:x_max,y_min:y_max]
        m=F.upsample_bilinear(m.unsqueeze(0),(patch_size,patch_size))
        resized_images.append(img)
        resized_masks.append(m)
    return torch.cat(resized_images,dim=0).cuda(),torch.cat(resized_masks,dim=0).cuda()




class SpatiallyWeightedMSE(_Loss):
    def __init__(self,rho=0.99,is_cuda=True):
        super(SpatiallyWeightedMSE, self).__init__()
        self.rho =rho
        self.is_cuda=is_cuda

    def find_weight(self,mask,rho=0.90):
        weight=np.zeros(mask.shape) 
        center,coords = find_center(mask[0])
        x_min,x_max,y_min,y_max=coords
        for x in range(mask[0].shape[0]):
            for y in range(mask[0].shape[1]):
                if mask[0][x][y]!=0:
                    distance=min([abs(x-x_min),abs(y-y_min),abs(x-x_max),abs(y-y_max)])
                    weight[0][x][y]=rho**distance
                else:
                    weight[0][x][y]=0
        return torch.Tensor(weight)
    
    def find_weights(self,masks,is_cuda=True):
        if is_cuda:
            masks=masks.cpu().numpy()
        else:
            masks=masks.numpy()
        weights=map(lambda x: self.find_weight(masks[x]).unsqueeze(0),range(masks.shape[0]))
        return torch.cat(list(weights),dim=0)

    def forward(self,input,target,masks):
        weights=self.find_weights(masks,self.is_cuda)
        loss=torch.mul(V(weights.cuda()),(input-target))**2
        return loss


import random
def random_mask(mask_size=(64,64),mask_size_range=(24,32)):
    w, h = mask_size
    th, tw = random.randint(mask_size_range[0],mask_size_range[1]),random.randint(mask_size_range[0],mask_size_range[1]) 
    if w == tw and h == th:
        return 0, 0, h, w

    i_min = random.randint(0, h - th)
    j_min = random.randint(0, w - tw)
    i_end=i_min+th
    j_end=j_min+tw
    # print (i_min,j_min,i_end,j_end)
    mask=np.zeros(mask_size)
    for x in range(w):
        for y in range(h):
            if x>=i_min and y>=j_min and x<=i_end and y<j_end:
                mask[x][y]=1
            else:
                mask[x][y]=0 
    mask=mask.reshape(1,1,mask.shape[0],mask.shape[1])
    return torch.Tensor(mask)


def random_masks(batch_size,size=(64,64),mask_size_range=[24,32]):
    random_masks=[]
    random_masks=map(lambda x: random_mask(),range(batch_size))
    return torch.cat(random_masks,dim=0)

def binary_encode(val,nb_digits=4):
    encoding=list(map(lambda x: int(x),('{0:0'+str(nb_digits)+'b}').format(val)))
    return val,np.array(encoding)

def encoding_tile(val,size_image=(64,64)):
    tile=np.tile(val[1],(size_image[0],size_image[1]//len(val[1])))
    return val[0],np.expand_dims(tile,axis=0)

def one_hot(val,nb_digits=10):
    label=[0]*nb_digits
    label[val]=1
    return np.array(label)



# from model import Classifier
# import torch.optim as optim

# train_loader = torch.utils.data.DataLoader(
#     dset.MNIST('data', train=True, download=True, transform=image_transform),
#     batch_size=batch_size,sampler=SubsetRandomSampler(train_indices))

# valid_loader = torch.utils.data.DataLoader(
#     dset.MNIST('data', train=True, download=True, transform=image_transform),
#     batch_size=batch_size,sampler=SubsetRandomSampler(valid_indices))

# criterion = nn.CrossEntropyLoss()

# # optimizer = optim.SGD(C.parameters(), lr=0.001, momentum=0.9)
# def train_classifier(train_loader,valid_loader,is_cuda,lr,epochs=5):
#     if is_cuda:
#         C=Classifier().cuda()
#     else:
#         C=Classifier()
#     C_optimizer=optim.Adam(C.parameters(), lr=lr)

#     for epoch in range(epochs):  # loop over the dataset multiple times

#         running_loss = 0.0
#         for i, data in tqdm.tqdm(enumerate(train_loader)):
#             # get the inputs
#             inputs, labels = data
#             # break
#             # wrap them in Variable
#             inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
#             # break
#             # zero the parameter gradients
#             C_optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = C(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             C_optimizer.step()

#             # print statistics
#             running_loss += loss.data[0]
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0

#         correct = 0
#         total = 0
#         for data in valid_loader:
#             # data.cuda()
#             images, labels = data
#             # break
#             outputs = C(Variable(images).cuda())
#             _, predicted = torch.max(outputs.data, 1)
#             # print (predicted.size())
#             total += labels.size(0)
#             correct += (predicted.cpu() == labels).sum()

#         print('Accuracy of the network on the 10000 test images: %d %%' % (
#             100 * correct / total))
#     print('Finished Training')
#     torch.save(C,open("Classifier_Acc_"+str(100 * correct / total)[:5]+".mdl","wb"))

import glob 
from PIL import Image,ImageDraw
class TestDataSet(Dataset):
    """docstring for TestDataSet"""
    def __init__(self,labels,image_transform,target_transform):
        super(TestDataSet, self).__init__()
        # self.images = glob.glob(data_loc+"/*")
        self.dataset=dset.MNIST('data', train=True, download=True, transform=image_transform,target_transform=label_transform)
        self.labels= labels
        # self.len=num_of_images
        self.image_transform=image_transform
        self.target_transform=target_transform
    def __len__(self):
        return  len(self.dataset)

    def __getitem__(self,index):
        return self.dataset.__getitem__(index)

# dataset=dset.MNIST('data', train=True, download=True, transform=image_transform,target_transform=label_transform)
# t=TestDataSet("trainingSample/",5,[1,2,3,4,5],image_transform=image_transform,target_transform=label_transform) 
def test(dataset,fake_labels,patch="small"):
    G=torch.load(open("models/Generator_v2_v2b256lr5e-4islblTepoc1000lrdecreaseper15_55_Namespace(batch_size=256, depochs=5, epochs=100, exp_name='v2b256lr5e-4islblTepoc1000lrdecreaseper15', gepochs=1, is_cuda=True, is_label_added=True, lr=0.0005, resume=False).mdl","rb")).cpu()
    new_im = Image.new('L', (64*2,128*len(fake_labels)))
    for ind,fk_lbl in enumerate(fake_labels):
        val=dataset.__getitem__(ind)
        img_r,label,category=val[0],val[1][1],val[1][0]
        if patch=='small':
            img_c,masks=preprocess(img_r)
        else:
            img_c,masks=create_mask(img_r.unsqueeze(0))
        Ginput=torch.cat([img_c,masks,torch.Tensor(label).unsqueeze(0).float()],dim=1) 
        op=G(V(Ginput)) 
        temp=img_c.clone()                          
        temp[temp==0.13]=0                         
        img_inp=((op.data*masks)+temp)
        img=torch.zeros([1,128,64]) 
        img[0,0:64,0:64]=img_inp[0] 
        img[0,64:128,0:64]=img_c  
        output=torchvision.transforms.ToPILImage()(img)
        d = ImageDraw.Draw(output)
        d.text((2,2),"Match"+str(category), fill=(255))
        # output.show()
        new_im.paste(output,(0,ind*128))
        fake_label=torch.Tensor(encoding_tile(binary_encode(fk_lbl))[1])    
        Ginput=torch.cat([img_c,masks,torch.Tensor(fake_label).unsqueeze(0).float()],dim=1) 
        op=G(V(Ginput)) 
        temp=img_c.clone()                          
        temp[temp==0.13]=0                         
        img_inp=((op.data*masks)+temp)
        img=torch.zeros([1,128,64]) 
        img[0,0:64,0:64]=img_inp[0] 
        img[0,64:128,0:64]=img_r         
        output=torchvision.transforms.ToPILImage()(img)
        d = ImageDraw.Draw(output)
        d.text((2,2),"MisMatch"+str(fk_lbl), fill=(255))
        # output.show()
        new_im.paste(output,(64,ind*128))
    new_im.show()


def create_mask(img,mean=0.13,type="large"):

    masks=np.zeros(img.size())
    c=random.choice([0,1,2,3])
    if c==0:
        x_s,x_e,y_s,y_e=0,32,0,64
    if c==1:
        x_s,x_e,y_s,y_e=32,64,0,64
    if c==2:
        x_s,x_e,y_s,y_e=0,64,0,32
    if c==3:    
        x_s,x_e,y_s,y_e=0,64,32,64
    masks[:,:,x_s:x_e,y_s:y_e]=1    
    masks=torch.Tensor(masks)
    img_c=torch.mul(img,(1-masks))
    img_c[masks==1]=mean
    return img_c,masks

    for val in t:
        img_r,label,category=val[0],val[1][1],val[1][0]
        img_c,masks=create_mask(img_r.unsqueeze(0))
        Ginput=torch.cat([img_c,masks,torch.Tensor(label).unsqueeze(0).float()],dim=1) 
        op=G(V(Ginput)) 
        temp=img_c.clone()                          
        temp[temp==0.13]=0
        img_inp=((op.data*masks)+temp) 
        img=torch.zeros([1,128,64]) 
        img[0,0:64,0:64]=img_inp[0] 
        img[0,64:128,0:64]=img_r 
        # x=torch.cat([img_r,img_inp[0]],axis=1)
        # print(x.size())                         
        output=torchvision.transforms.ToPILImage()(img)
        d = ImageDraw.Draw(output)
        d.text((2,2), str(category), fill=(255))
        output.show()








