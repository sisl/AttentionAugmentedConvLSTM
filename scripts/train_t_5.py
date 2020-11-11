"""
Code to train PredNet with TAA/SAAConvLSTM in t+5 mode
Ground truth is provided sequentailly extrap_start_time=5.
Then the prediction is provided instead.
Follows the procedure defined by Lotter et al.

Based on the existing implementations:
- Itkina, Masha, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. Tensorflow
implementation of "Dynamic Environment Prediction in Urban Scenes using Recurrent Representation Learning."
https://github.com/mitkina/EnvironmentPrediction,
-Bill Lotter, Gabriel Kreiman, and David Cox Official PredNet Tensorflow
implementation: https://github.com/coxlab/prednet,
- Leido Pytorch implementation of PredNet:
https://github.com/leido/pytorch-prednet/blob/master/kitti_data.py".
"""

import os
import numpy as np
import random as rn
import hickle as hkl
import argparse
import yaml
import sys

sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.model import Model
from model.kitti_data import KITTI



seed = 123
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CONFIG_PATH = ""

def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 5:, :, :, :]
    y_hat = y_hat[:, 5:, :, :, :]
    return 0.5 * torch.mean(torch.abs(y_true - y_hat)).cuda()

def lr_scheduler(optimizer, epoch, num_iter):
    lr = 0.00035 * min(0.0002*num_iter,np.exp(-0.00005*num_iter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    writer.add_scalar('Learning Rate', lr, num_iter)

    return optimizer, lr

def lr_scheduler_new(optimizer, epoch, num_iter):
    if epoch < 75:
        for param_group in optimizer.param_groups:
            lr = 0.0001
            param_group['lr'] = lr
            writer.add_scalar('Learning Rate', lr, num_iter)
        return optimizer, lr
    else:
        for param_group in optimizer.param_groups:
            lr = 0.00001
            param_group['lr'] = lr
            writer.add_scalar('Learning Rate', lr, num_iter)
        return optimizer, lr

def gradient_vis(parameters, num_inter):
        for name, param in parameters:
            if param.requires_grad and param.grad!=None:
                writer.add_scalar('Gradients/'+str(name)+'_avg', param.grad.abs().mean(), num_inter)
                writer.add_scalar('Gradients/'+str(name)+'_max', param.grad.abs().max(), num_inter)
            elif(param.grad==None):
                print(name)

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


parser = argparse.ArgumentParser()
parser.add_argument('config_filename')
args = parser.parse_args()
config = load_config(args.config_filename)

#Setting up the model
nt = config["model"]["nt"]
nb_layers = config["model"]["nb_of_layers"]
(n_channels, im_height, im_width) = tuple(config["model"]["input_shape"])
input_shape = (im_height, im_width, n_channels)
stack_sizes = (n_channels, *tuple(config["model"]["stack_sizes"]))
R_stack_sizes = stack_sizes
A_filt_sizes = tuple(config["model"]["A_filt_sizes"])
Ahat_filt_sizes = tuple(config["model"]["Ahat_filt_sizes"])
R_filt_sizes = tuple(config["model"]["R_filt_sizes"])
layers_type = config["model"]["type_of_all_layers"]
attention_horizon = config["model"]["attention_horizon"]
Nh = config["model"]["Nh"]
dk = config["model"]["key_query_dimension"]
dv = config["model"]["value_dimension"]

#Setting up training
model_name = config["training"]["model_name"]
data_directory = config["training"]["dataset_path"]
nb_epoch = config["training"]["nb_epochs"]
batch_size = config["training"]["batch_size"]
samples_per_epoch =  config["training"]["samples_per_epoch"]
N_seq_val = config["training"]["N_seq_val"]
save_model = config["training"]["save_model"]

#Loading the dataset
train_file = os.path.join(data_directory, 'X_train.hkl')
train_sources = os.path.join(data_directory, 'sources_train.hkl')
val_file = os.path.join(data_directory, 'X_val.hkl')
val_sources = os.path.join(data_directory, 'sources_val.hkl')
kitti_train = KITTI(train_file, train_sources, nt)
kitti_val = KITTI(val_file, val_sources, nt)

model = Model(layers_type, stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, attention_horizon,
                  dk, dv, Nh, im_width, im_height, output_mode ='prediction',
                  extrap_start_time = 5, positional_encoding= True, forget_bias=1.0)

checkpoint = torch.load("models/{}/{}_t+1.pt".format(model_name, model_name))
model.load_state_dict(checkpoint['model_state_dict'])

torch.cuda.empty_cache()
if torch.cuda.is_available():
    print('Using GPU.')
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.98), eps=1e-09)
train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
writer = SummaryWriter("../trained_models/{}/{}_t+5_tensorboard".format(model_name, model_name))
num_iter, loss_train, loss_val, loss_train_iter, loss_val_iter = 0, 0, 0, 0, 0
val_min = True
loss_val_min = float('Inf')
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.98), eps=1e-09)

print("Training: {}_t+5".format(model_name))

for epoch in range(nb_epoch):

    loss_per_epoch = 0
    iter_per_epoch = 0

    for i, inputs in enumerate(train_loader):
        if(i<samples_per_epoch):

            optimizer, lr = lr_scheduler_new(optimizer, epoch, num_iter)
            inputs = inputs.permute(0, 1, 4, 2, 3)
            inputs = inputs.cuda()
            y_hat = model(inputs) # batch x n_layers x nt
            errors = extrap_loss(y_hat, inputs)
            optimizer.zero_grad()
            errors.backward()

            if(num_iter%20==0):
                gradient_vis(model.named_parameters(), num_iter)
            optimizer.step()

            loss_train = loss_train + errors.detach()
            loss_train_iter += 1
            num_iter += 1
        else:
            print('Epoch: {}/{}, step: {}/{}, loss: {}'.format(epoch, nb_epoch, i, len(kitti_train)//batch_size,loss_train/loss_train_iter))
            writer.add_scalar('Loss/train', loss_train/loss_train_iter, num_iter)

            loss_train = 0
            loss_train_iter = 0

            with torch.no_grad():
                for j, inputs in enumerate(val_loader):
                    if(j<N_seq_val):
                        inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
                        inputs = inputs.cuda()
                        y_hat = model(inputs) # batch x n_layers x nt
                        errors = extrap_loss(y_hat, inputs)
                        loss_val = loss_val + errors.detach()
                        loss_val_iter += 1

                if loss_val/loss_val_iter < loss_val_min:
                    loss_val_min = loss_val/loss_val_iter
                    val_min = True

                writer.add_scalar('Loss/val', loss_val/loss_val_iter, num_iter)
                loss_val = 0
                loss_val_iter = 0
                num_iter += 1
            break


    if val_min == True or epoch%100 == 0:
        print("Saving the best model at epoch {}".format(epoch))
        val_min = False
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': errors,
            }, "models/{}/{}_t+5_epoch_{}.pt".format(model_name, model_name, epoch))

if save_model:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': errors,
        }, "../trained_models/{}/{}_t+5.pt".format(model_name, model_name))

writer.close()
