import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as scio
import random
'''Building Blocks'''
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_Size, stride=1, padding=(2,2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_Size, stride=2, padding=(2,2))
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_Size, stride=1, padding=(2,2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_Size, stride=1, padding=(2,2))
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(down, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch, kernel_Size)
        self.inconv = inconv(out_ch, out_ch, kernel_Size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_Size, stride=2, padding=(2,2))
        self.dropout = nn.Dropout(p=0.02)
#            nn.MaxPool2d(2), # this kernel size could be changed       

    def forward(self, x):
        x1 = self.mpconv(x)
        x2 = self.conv(x)
        x = torch.add(x1,x2)
        x = self.dropout(x)
        x3 = self.inconv(x)
        x = torch.add(x,x3)
        x = self.dropout(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, cat_ch, out_ch, kernel_size, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride=2, padding=(2,2))

        self.conv = inconv(in_ch+cat_ch, out_ch, kernel_size)
        self.BN1 = nn.BatchNorm2d(in_ch)
        self.BN2 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, x0, x2):
        x1 = self.BN1(x0)
        x1 = self.ReLU(x1)
        x1 = self.up(x1)
        x1 = self.BN1(x1)
        x1 = self.ReLU(x1)
        x3 = self.up(x0)
        x1 = torch.add(x1,x3)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
      #  diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        #                diffZ // 2, diffZ - diffZ//2)) # not sure it is correct

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_Size):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_Size, stride=1, padding=(2,2))
        self.inconv = inconv(in_ch, out_ch, kernel_Size)
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, x):
        x2 = self.inconv(x)
        x3 = self.conv(x)
        x = torch.add(x2,x3)
        x = self.dropout(x)
        return x



'''loss function'''
class npcc_lossFunc(nn.Module):
	def _init_(self):
		super(npcc_lossFunc,self).__init__()
		return

	def forward(self, G, T):

		fsp = G - torch.mean(G,0)
		fst = T - torch.mean(T,0)
		devP = torch.std(G,0,True,False)
		devT = torch.std(T,0,True, False)
    
		npcc_loss=(-1)*torch.mean(fsp*fst,0)/torch.clamp(devP*devT,1e-7,None)
		npcc_loss=torch.mean(npcc_loss,0)
		return npcc_loss

	def backward(self,grad_output):
		return grad_output


'''data loading'''
def CreateDataset(Train_idxes,Test_idxes,str1,str2):
    print('Loading dataset')
    paths_list_1 = os.listdir(str1)
    paths_list_2 = os.listdir(str2)
    Train_dataset = []
    for idx in Train_idxes:
        # print(idx)
        inputpath = str1 + '\\' + paths_list_1[idx]
        gtpath = str2 +  '\\' + paths_list_2[idx]
        if (os.path.exists(inputpath)) is False or (os.path.exists(gtpath)) is False:
            return -1
        else:
            # print(inputpath)
            # print(gtpath)
            Inputs = scio.loadmat(inputpath) #input_path
            Phi1 = Inputs['Phi_crude'] # load approximant
            img = np.transpose(Phi1,(2,0,1)) # (4, 256, 256)
            Outputs = scio.loadmat(gtpath) #output_path
            dn = (Outputs['n_pred']-1.337)*100
            gt = np.transpose(dn,(2,0,1)) # (100, 256, 256)
            Train_dataset.append([img,gt])
    Test_dataset = []
    for ii in Test_idxes:
        # print(ii)
        inputpath = str1 +  '\\' + paths_list_1[ii]
        gtpath = str2 +  '\\' + paths_list_2[ii]
        if (os.path.exists(inputpath)) is False or (os.path.exists(gtpath)) is False:
            return -1
        else:
            # print(inputpath)
            # print(gtpath)
            Inputs = scio.loadmat(inputpath) #input_path
            Phi2 = Inputs['Phi_crude']
            img1 = np.transpose(Phi2,(2,0,1)) # (4, 256, 256)
            Outputs = scio.loadmat(gtpath) #output_path
            dn = (Outputs['n_pred']-1.337)*100
            gt1 = np.transpose(dn,(2,0,1)) # (100, 256, 256)
            Test_dataset.append([img1,gt1])
    return Train_dataset, Test_dataset

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b