from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import os,sys
import time

# Normal UResNet and sparse version
from sparse_uresnet.model import SparseUResNet
#
# from sparsio.iotools import io_factory

# Accelerate *if all input sizes are same*
torch.backends.cudnn.benchmark = True

class trainval(object):

    def __init__(self,flags):
        self._flags=flags
        self.tspent_save  = -1.
        self.tspent_train = -1.
        self.tspent_inference = -1.
        
    def initialize(self):
        self._net = SparseUResNet(True,
                                  num_strides=self._flags.NUM_STRIDES,
                                  base_num_outputs=self._flags.BASE_NUM_OUTPUTS)
        print(self._net)
        # Define loss and optimizer
        if self._flags.TRAIN:
            self._criterion = nn.CrossEntropyLoss()
            self._optimizer = optim.Adam(self._net.parameters(), lr=self._flags.LEARNING_RATE)
            self._lrop      = optim.lr_scheduler.StepLR(self._optimizer, 1000, gamma=0.1)
            print(os.environ['CUDA_VISIBLE_DEVICES'])
            self._net.train().cuda()
        else:
            self._softmax = nn.LogSoftmax(dim=1)
            self._net.eval().cuda()

        iteration = 0
        if self._flags.INPUT_WEIGHT:
            if not os.path.isfile(self._flags.INPUT_WEIGHT):
                sys.stderr.write('File not found: %s\n' % self._flags.INPUT_WEIGHT)
                raise ValueError
            print('Restoring weights from %s...' % self._flags.INPUT_WEIGHT)
            with open(self._flags.INPUT_WEIGHT, 'rb') as f:
                checkpoint = torch.load(f)
                self._net.load_state_dict(checkpoint['state'])
                # print(checkpoint['state_dict'])
                if self._flags.TRAIN:
                    self._optimizer.load_state_dict(checkpoint['optimizer'])
                    self._lrop.load_state_dict(checkpoint['lrop'])
                    iteration = checkpoint['iteration'] + 1
            print('Done.')
        return iteration

    def save_state(self,iteration):
        tstart = time.time()
        
        filename = '%s-%d.ckpt' % (self._flags.SAVE_WEIGHT,iteration)
        torch.save({
            'iteration': iteration,
            'state': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'lrop': self._lrop.state_dict()
        }, filename)        
        self.tspent_save = time.time() - tstart
        
    def train_step(self,voxel,feature,label):
        if not self._flags.TRAIN:
            return
        tstart = time.time()
        
        # Forward
        coords      = torch.from_numpy(voxel).cuda()
        features    = torch.from_numpy(feature).cuda()
        predictions_raw = self._net(coords, features)

        # Loss & apply gradient
        label_vals = torch.from_numpy(label).cuda().type(torch.cuda.LongTensor)
        loss_vals  = self._criterion(predictions_raw, label_vals)
        # lr_scheduler.step()  # Decay learning rate
        self._optimizer.zero_grad()  # Clear previous gradients
        loss_vals.backward()  # Compute gradients of all variables wrt loss
        nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)  # Clip gradient
        self._optimizer.step()  # update using computed gradients
        self.tspent_train = time.time() - tstart
        
        # Accuracy
        predicted_labels = torch.argmax(predictions_raw, dim=1)
        #acc = (predicted_labels == label_vals).sum().item() / float(label_vals.nelement())
        #return loss_vals.mean(),acc

        return loss_vals.mean(),predicted_labels.cpu().detach().numpy()

    def inference_step(self,voxel,feature):
        if self._flags.TRAIN:
            return
        tstart = time.time()
        
        # Forward
        coords      = torch.from_numpy(voxel).cuda()
        features    = torch.from_numpy(feature).cuda()
        predictions_raw = self._net(coords, features)
        
        self.tspent_inference = time.time() - tstart
        return self._softmax(predictions_raw).cpu().detach().numpy()
