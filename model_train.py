import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from glob import glob
import random
import numpy as np
from PIL import Image
from utils import Data
from torch.autograd import Variable as V
import torch.nn as nn
import torch.optim as optim
from utils import find_patches
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
import numpy as np


def one_hot(val,nb_digits=10):
	label=[0]*nb_digits
	label[val]=1
	return np.array(label)

batch_size=16
NUM_OF_EPOCHs=50
options={"clamp_lower":-0.01,
		"clamp_upper":0.01}
W_NOISE=False
one = torch.FloatTensor([1])
mone = one * -1
IS_CUDA=True
RESUME=True

img_size = 28
image_transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor()
])
label_transform=transforms.Compose([lambda x: one_hot(x)])


train_loader = torch.utils.data.DataLoader(
    dset.MNIST('data', train=True, download=True, transform=image_transform,target_transform=label_transform),
    batch_size=batch_size, shuffle=True)


