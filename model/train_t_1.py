import os
import numpy as np
import argparse
import yaml

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import Model
from kitti_data import KITTI

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def lr_scheduler_new(optimizer, num_iter):
    lr = 0.0005 * min(0.0002*num_iter,np.exp(-0.00005*num_iter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('Learning Rate', lr, num_iter)

    return optimizer

def gradient_vis(parameters, num_inter):
        for name, param in parameters:
            if param.requires_grad and param.grad!=None:
                writer.add_scalar('Gradients/'+str(name)+'_avg', param.grad.abs().mean(), num_inter)
                writer.add_scalar('Gradients/'+str(name)+'_max', param.grad.abs().max(), num_inter)
            elif(param.grad==None):
                print(name)

def load_config(config_name):
    CONFIG_PATH = ""
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
layer_type = config["model"]["type_of_all_layers"]
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

#Loading the dataset
train_file = os.path.join(data_directory, 'X_train.hkl')
train_sources = os.path.join(data_directory, 'sources_train.hkl')
val_file = os.path.join(data_directory, 'X_val.hkl')
val_sources = os.path.join(data_directory, 'sources_val.hkl')
kitti_train = KITTI(train_file, train_sources, nt)
kitti_val = KITTI(val_file, val_sources, nt)

#Setting up loss weights
layer_loss_weights = Variable(torch.FloatTensor([[1.], *[[0.]]*(nb_layers-1)])).cuda()
time_loss_weights = 1./ (nt - 5) * torch.ones(nt,1).cuda()
time_loss_weights[0:5] = 0
time_loss_weights = Variable(time_loss_weights).cuda()

print("Training: {}_t+1".format(model_name))
writer = SummaryWriter("models/{}/{}_t+1_tensorboard".format(model_name, model_name))
model = Model(layer_type, stack_sizes, R_stack_sizes,
                            A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, attention_horizon,
                            dk, dv, Nh, im_width, im_height, positional_encoding= True, forget_bias=1.0)

torch.cuda.empty_cache()
if torch.cuda.is_available():
    print('Using GPU.')
    model = model.cuda()

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

num_iter, loss_train, loss_val, loss_train_iter, loss_val_iter = 0, 0, 0, 0, 0
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.98), eps=1e-09)

for epoch in range(nb_epoch):
    loss_per_epoch = 0
    iter_per_epoch = 0
    for i, inputs in enumerate(train_loader):
        if(i<samples_per_epoch):
            optimizer = lr_scheduler_new(optimizer, num_iter)
            inputs = inputs.permute(0, 1, 4, 2, 3)
            inputs = inputs.cuda()
            errors = model(inputs)
            errors = errors.float()
            loc_batch = errors.size(0)
            errors = torch.mm(errors.view(-1, nt), time_loss_weights)
            errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
            errors = torch.mean(errors)
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
                        errors = model(inputs) # batch x n_layers x nt
                        errors = errors.float()
                        loc_batch = errors.size(0)
                        errors = torch.mm(errors.view(-1, nt), time_loss_weights)
                        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
                        errors = torch.mean(errors)
                        loss_val = loss_val + errors.detach()
                        loss_val_iter += 1

                writer.add_scalar('Loss/val', loss_val/loss_val_iter, num_iter)
                loss_val = 0
                loss_val_iter = 0
                num_iter += 1
            break


torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': errors,
    }, "models/{}/{}_t+1.pt".format(model_name, model_name))
writer.close()
