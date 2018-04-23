import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from glob import glob
import random
import numpy as np
from PIL import Image
from torch.autograd import Variable as V
import torch.nn as nn
import torch.optim as optim
from utils import find_patches
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from utils import *
from model import Generator,Discriminator
import torchvision.datasets as dset
import tqdm
import argparse



options={"clamp_lower":-0.01,
		"clamp_upper":0.01}


parser = argparse.ArgumentParser()
parser.add_argument("-is_cuda", default=True,help="is_cuda",type=bool)
parser.add_argument("-resume", default=False,help="resume",type=bool)
parser.add_argument("-is_label_added", default=True,help="is_label_added",type=bool)
parser.add_argument("-epochs", default=26,help="number of epochs",type=int)
parser.add_argument("-gepochs", default=1,help="number of g epochs",type=int)
parser.add_argument("-depochs", default=5,help="number of d epochs",type=int)
parser.add_argument("-batch_size", default=128,help="batch_size",type=int)
parser.add_argument("-lr", default=5e-5,help="learning rate",type=float)
parser.add_argument("-exp_name", help="experiment_name",type=str)


args = parser.parse_args()


batch_size=128
W_NOISE=False
IS_CUDA=args.is_cuda
RESUME=args.resume
IS_LABEL_ADDED=args.is_label_added
img_size = 64
lr=args.lr
NUM_OF_EPOCHs=args.epochs
G_PRETRAIN=args.gepochs
D_PRETRAIN=args.depochs



image_transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor()
])

writer = SummaryWriter(args.exp_name)
if IS_LABEL_ADDED:
	label_transform=transforms.Compose([lambda x: binary_encode(x),lambda x: encoding_tile(x)])
	G=Generator(in_c=3).cuda()
	D=Discriminator(in_c=2).cuda()

else:	
	label_transform=transforms.Compose([lambda x: one_hot(x)])
	G=Generator().cuda()
	D=Discriminator().cuda()


train_indices=range(0,59967)
valid_indices=range(59968,60000)

train_loader = torch.utils.data.DataLoader(
    dset.MNIST('data', train=True, download=True, transform=image_transform,target_transform=label_transform),
    batch_size=batch_size,sampler=SubsetRandomSampler(train_indices))

valid_loader = torch.utils.data.DataLoader(
    dset.MNIST('data', train=True, download=True, transform=image_transform,target_transform=label_transform),
    batch_size=batch_size,sampler=SubsetRandomSampler(valid_indices))
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

def reset_grad():
    G.zero_grad()
    D.zero_grad()


number_of_batches=len(train_loader)


reconLoss=SpatiallyWeightedMSE()


def preprocess(img,mean=0.13):
	masks=random_masks(img.size()[0])			
	img_c=torch.mul(img,(1-masks))
	img_c[masks==1]=mean
	return img_c,masks


def train_discriminator(D,D_optimizer,val,G,writer,epoch):
	LAMBDA=0.01
	try:
		img_r,label=val[0],val[1][1]
		# img_r,label=val
		img_c,masks=preprocess(img_r)
		if IS_CUDA:
			img_r,img_c,masks,label=img_r.cuda(),img_c.cuda(),masks.cuda(),label.cuda()


		#train with real
		if not IS_LABEL_ADDED:
			Dinput=img_r
		else:
			Dinput=torch.cat([img_r,label.float()],dim=1)
		errD_real=D(V(Dinput),masks)

		#train with fake
		if IS_LABEL_ADDED:
			Ginput=torch.cat([img_c,masks,label.float()],dim=1)
		else:
			Ginput=torch.cat([img_c,masks],dim=1)
		op=G(V(Ginput))
		temp=img_c.clone()
		temp[temp==0.13]=0
		img_inp=((op*V(masks))+V(temp))
		if not IS_LABEL_ADDED:
			Dinput=img_inp.clone()
		else:
			Dinput=torch.cat([img_inp,V(masks)],dim=1).clone()
		errD_fake=D(Dinput,masks).mean()
		#Gradient Penalty
		# alpha=random.random()
		# x_hat = (alpha*img_inp+(1-alpha)*V(img_r)).detach()
		# x_hat.requires_grad = True
		# loss_D = D(x_hat,masks).sum()
		# loss_D.backward()
		# x_hat.grad.volatile = False			
		# Penalty=(torch.mul(V(masks),(x_hat.grad -1)**2 * LAMBDA)).mean()

		errD=-(errD_real-errD_fake).mean()#+Penalty
		errD.backward()


		D_optimizer.step()
		reset_grad()
		writer.add_scalar("Discriminator/Error",errD,epoch)
		writer.export_scalars_to_json("./all_scalars.json")
		# weight clipping
		for p in D.parameters():
			p.data.clamp_(options["clamp_lower"],options["clamp_upper"])
	except Exception as e:
		print (e)

def train_generator(G,G_optimizer,val,D,C,writer,epoch,both_loss=False,delta=1e-3,rho=1e-3):
	class_criterion = nn.CrossEntropyLoss()
	try:
		img_r,label,category=val[0],val[1][1],val[1][0]
		# img_r,label=val
		img_c,masks=preprocess(img_r)
		if IS_CUDA:
			img_r,img_c,masks,label,category=img_r.cuda(),img_c.cuda(),masks.cuda(),label.cuda(),category.cuda()
		if IS_LABEL_ADDED:
			Ginput=torch.cat([img_c,masks,label.float()],dim=1)
		else:
			Ginput=torch.cat([img_c,masks],dim=1)
		op=G(V(Ginput))
		temp=img_c.clone()
		temp[temp==0.13]=0
		img_inp=((op*V(masks))+V(temp))
		if not IS_LABEL_ADDED:
			Dinput=img_inp
		else:
			Dinput=torch.cat([img_inp,V(masks)],dim=1)
		D_score=D(Dinput,masks).mean()
		
		errG_adv=-(D_score.mean())

		recon_loss=reconLoss(img_inp,V(img_r),masks).mean()
		# print(C(img_inp).size(),category.size())
		if C:
			classification_loss=class_criterion(C(img_inp),V(category))


		# break
		# recon_loss.backward()
		# print("Recon",recon_loss.data,"Class",classification_loss.data,"Adv",errG_adv.data)
		if both_loss:
			# errG=delta*errG_adv
			if  C:
				errG=delta*errG_adv+recon_loss+rho*classification_loss
			else:
				errG=delta*errG_adv+recon_loss
		else:
			if  C:
				errG=recon_loss+rho*classification_loss
			else:
				errG=recon_loss
		errG.backward()
		G_optimizer.step()
		reset_grad()
		writer.add_scalar("Generator/Error",errG,epoch)
		writer.export_scalars_to_json("./all_scalars.json")
	except Exception as e:
		print (e)


def validate(G,valid_dataloader,writer,epoch):
	flg=0
	G.eval()
	for index,val in enumerate(valid_dataloader):
		try:
			img_r,label,category=val[0],val[1][1],val[1][0]
			# img_r,label=val
			img_c,masks=preprocess(img_r)
			if IS_CUDA:
				img_r,img_c,masks,label=img_r.cuda(),img_c.cuda(),masks.cuda(),label.cuda()
			# break
			if IS_LABEL_ADDED:
				Ginput=torch.cat([img_c,masks,label.float()],dim=1)
			else:
				Ginput=torch.cat([img_c,masks],dim=1)
			op=G(V(Ginput))
			temp=img_c.clone()
			temp[temp==0.13]=0			
			img_inp=((op.data*masks)+temp)
			if flg==0:
				result=img_inp
				real_image=img_r
				input=img_c
				flg=1
			else:
				result=torch.cat((result,img_inp),dim=0)
				real_image=torch.cat((real_image,img_r),dim=0)
				input=torch.cat((input,img_c),dim=0)
		except Exception as e:
			print (e)
			continue
	G.train()
	result=vutils.make_grid(result,nrow=5)
	gt=vutils.make_grid(real_image,nrow=5)
	inp=vutils.make_grid(input,nrow=5)
	writer.add_image("Validation Images/Inpainted",result,epoch)
	writer.add_image("Validation Images/GroundTruth",gt,epoch)
	writer.add_image("Validation Images/Input",inp,epoch)
	writer.export_scalars_to_json("./all_scalars.json")


from model import Classifier

C=torch.load(open("Classifier_Acc_100.0.mdl","rb"))



for epoch in range(NUM_OF_EPOCHs):
	if epoch%15==0:
			lr=lr*0.1
			G_optimizer = optim.Adam(G.parameters(), lr=lr)
			D_optimizer = optim.Adam(D.parameters(), lr=lr)
	if epoch%5==0:
		validate(G,valid_loader,writer,epoch)
		torch.save(D,open("models/Discriminator_v2_"+args.exp_name+"_"+"_"+str(epoch)+"_"+str(args)+".mdl","wb"))		
		torch.save(G,open("models/Generator_v2_"+args.exp_name+"_"+str(epoch)+"_"+str(args)+".mdl","wb"))		
	ind=0
	for val in tqdm.tqdm(train_loader):
		# k=val 
		# break
		ind+=1
		e=number_of_batches*epoch+ind
		if epoch<G_PRETRAIN:
			train_generator(G,G_optimizer,val,D,C,writer,e,both_loss=False)
		elif epoch<(G_PRETRAIN+D_PRETRAIN):
			train_discriminator(D,D_optimizer,val,G,writer,e)
		else:
			train_generator(G,G_optimizer,val,D,C,writer,e,both_loss=True)
			for i in range(3):
				train_discriminator(D,D_optimizer,val,G,writer,e)


