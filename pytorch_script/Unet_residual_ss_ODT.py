import sys
import os
from optparse import OptionParser
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch import optim
from torch.autograd import Variable as var
from model import UNet_3
from util import npcc_lossFunc, CreateDataset, split_train_val, batch


'''Training'''
def train_net(net, epochs, batch_size, lr, val_percent, save_cp, gpu):# training function
# import data
    str1 = 'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\Deep-learning_ODT\\Training_Dataset\\Inputs_3'
    str2 = 'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\Deep-learning_ODT\\Training_Dataset\\GT_BPM_3'
    Train_idx = range(1,900)#range(1,900)
    Test_idx = range(901,939)
    [Train_dataset, Test_dataset] = CreateDataset(Train_idx,Test_idx,str1,str2)
    iddataset = split_train_val(Train_dataset, val_percent) # shuffle the data
    train = iddataset['train']
    val = iddataset['val']
# make direction for training results
    resultFolder = 'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\Deep-learning_ODT\\pytorch_script\\pred_result\\Pred_low_freq'
    if (os.path.exists(resultFolder)) is False:
        os.makedirs(resultFolder)
    else:
        pass

    print('''
        Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))
    # optimizer and loss functions
    # N_train = len(iddataset['train'])
    # optimizer_SGD = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = 0.0005)
    optimizer_adam = optim.Adam(net.parameters(), lr=lr, betas=[0.9,0.999], weight_decay = 0.0)
    criterion = npcc_lossFunc()
    if gpu:
        criterion.cuda()
    
    train_loss = []
    val_loss = []
    test_loss = []
    valid_loss_min = 0.00
    print('Start training')
    for epoch in range(epochs):
###  training
        net.train()
        train_epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            gts = np.array([i[1] for i in b]).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            gts = torch.from_numpy(gts)
            if gpu:
                imgs = imgs.cuda()
                gts = gts.cuda()
            gts_pred = net(imgs)
            gts_pred_flat = gts_pred.view(-1)
            gts_flat = gts.view(-1)

            loss = criterion(gts_pred_flat, gts_flat)
            train_epoch_loss += loss.item()

            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
        train_epoch_loss = train_epoch_loss/i
        train_loss.append(train_epoch_loss)
### validation
        net.eval()
        val_epoch_loss = 0

        for i, b in enumerate(batch(val, 1)): # batch_size =1
            img = np.array([i[0] for i in b]).astype(np.float32)
            gt = np.array([i[1] for i in b]).astype(np.float32)
            img = torch.from_numpy(img)
            gt = torch.from_numpy(gt)
            if gpu:
                img = img.cuda()
                gt = gt.cuda()
            gt_pred = net(img)
            gt_pred_flat = gt_pred.view(-1)
            gt_flat = gt.view(-1)

            loss = criterion(gt_pred_flat, gt_flat)
            val_epoch_loss += loss.item()
        val_epoch_loss = val_epoch_loss/i
        print('Epoch {}/{}, Training Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch + 1, epochs, train_epoch_loss,val_epoch_loss))
        val_loss.append(val_epoch_loss)
    ## save the model if validation loss has decreased
        if val_epoch_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_epoch_loss))
            valid_loss_min = val_epoch_loss
            netFile = resultFolder + '\\net_'+str(epoch)+'.pth'
            torch.save(net,netFile)


# Testing
    net.eval()
    test_loss = 0
    idx = 0

    for j, b in enumerate(batch(Test_dataset,1)):
        img1 = np.array([j[0] for j in b]).astype(np.float32)
        gt1 = np.array([j[1] for j in b]).astype(np.float32)
        img1 = torch.from_numpy(img1)
        gt1 = torch.from_numpy(gt1)
        idx += 1
        if gpu:
            img1 = img1.cuda()
            gt1 = gt1.cuda()
        
        gt_pred_1 = net(img1)
        gt1_pred = torch.Tensor.cpu(gt_pred_1)
        phi_pred = var(gt1_pred).numpy()
        testFile = resultFolder+'\\Prediction'+str(idx)+'.mat'
        scio.savemat(testFile, {'phi_pred':phi_pred})
        gt2 = torch.Tensor.cpu(gt1)
        gt2= var(gt2).numpy()
        gtFile = resultFolder+'\\GT_'+str(idx)+'.mat'
        scio.savemat(gtFile, {'gt':gt2})
        img2 = torch.Tensor.cpu(img1)
        img2= var(img2).numpy()
        inpFile = resultFolder+'\\Input_'+str(idx)+'.mat'
        scio.savemat(inpFile, {'inp':img2})

        gt1_pred_flat = gt_pred_1.view(-1)
        gt1_flat = gt1.view(-1)
        loss = criterion(gt1_pred_flat, gt1_flat)
        test_loss += loss.item()

    print('Testing Loss: {}'.format(test_loss/j))
    netFile = resultFolder + '\\net_DNN_L.pth'
    torch.save(net, netFile)

    y1 = np.array(train_loss)
    y2 = np.array(val_loss)

    NewFile_1 = resultFolder + '\\training_loss.mat'
    scio.savemat(NewFile_1, {'y1':y1})

    NewFile_2 = resultFolder + '\\validation_loss.mat'
    scio.savemat(NewFile_2, {'y2':y2})

    return train_loss, val_loss       

'''main file'''
if __name__ == '__main__':

    net = UNet_3(n_channels=4, n_classes=100)
    net.cuda()
    net = torch.nn.DataParallel(net)
    n_epoch = 500
    [train_loss,val_loss] = train_net(net=net, epochs = n_epoch, batch_size = 20, lr = 0.001, val_percent = 0.05, save_cp=True, gpu=True)

    x = np.arange(1,n_epoch+1)

    y1 = np.array(train_loss)
    y2 = np.array(val_loss)

    plt.plot(x, y1, color="r", linestyle="-", linewidth=1, label="training_loss")
    plt.plot(x, y2, color="b", linestyle="-", linewidth=1, label="validation_loss")

    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))

    plt.title("Training Curve")

    plt.savefig('low frequency Training curve.jpg')

    plt.show()







