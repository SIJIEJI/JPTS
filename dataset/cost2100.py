from cgi import test
import os
import numpy as np
import scipy.io as sio

import random
import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['Cost2100DataLoader', 'PreFetcher']


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class Cost2100DataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, scenario):
        assert os.path.isdir(root)
        assert scenario in {"in", "out"}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dir_train = os.path.join(root, f"DATA_Htrain{scenario}.mat")
        dir_val = os.path.join(root, f"DATA_Hval{scenario}.mat")
        dir_test = os.path.join(root, f"DATA_Htest{scenario}.mat")
        dir_raw = os.path.join(root, f"DATA_HtestF{scenario}_all.mat")
        channel, nt, nc, nc_expand = 2, 32, 32, 125

        # Training data loading
        data_train = sio.loadmat(dir_train)['HT']
        data_train = torch.tensor(data_train, dtype=torch.float32).view(
            data_train.shape[0], channel, nt, nc)

        
        suff = suffler(data_train) 
        aux_data_train,suff_idx = suff.suffle()
        train_data = torch.cat( (data_train,aux_data_train),1 ) 
        
        self.train_dataset = TensorDataset(train_data,suff_idx)

        data_val = sio.loadmat(dir_val)['HT']
        data_val = torch.tensor(data_val, dtype=torch.float32).view(
            data_val.shape[0], channel, nt, nc)
        
        self.val_dataset = TensorDataset(data_val)

        # Test data loading, including the sparse data and the raw data
        data_test = sio.loadmat(dir_test)['HT']
        data_test = torch.tensor(data_test, dtype=torch.float32).view(
            data_test.shape[0], channel, nt, nc)
        

        raw_test = sio.loadmat(dir_raw)['HF_all']
        real = torch.tensor(np.real(raw_test), dtype=torch.float32)
        imag = torch.tensor(np.imag(raw_test), dtype=torch.float32)
        raw_test = torch.cat((real.view(raw_test.shape[0], nt, nc_expand, 1),
                              imag.view(raw_test.shape[0], nt, nc_expand, 1)), dim=3)
        self.test_dataset = TensorDataset(data_test, raw_test)

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
                      
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            val_loader = PreFetcher(val_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, val_loader, test_loader
    
    def gen_sequence(self, max_val):
        seq = []
        while True:
            x = random.randint(0,max_val - 1)
            if x not in seq:
                seq.append(x)
            if len(seq) == max_val:
                break
        return seq

class suffler():
    def __init__(self, input_data, seed=random.random()):
        self.origin = input_data
        self.seed = seed
    
    def gen_sequence(self, max_val):
        seq = []
        while True:
            x = random.randint(0,max_val - 1)
            if x not in seq:
                seq.append(x)
            if len(seq) == max_val:
                break
        return seq

    def suffle(self):
        random.seed(self.seed)
        idx = []
        idx.extend([[0,16,0,16],  [0,16,16,32],
                    [16,32,0,16], [16,32,16,32]])
        s = torch.zeros((self.origin.size(0),2,32,32)) 
        
        suff_idx = torch.zeros(self.origin.size(0),4) 
       
        for i in range(0, self.origin.size(0)):
            seq = self.gen_sequence(len(idx)) 
            
            seq_idx = torch.tensor(seq)
            suff_idx[i,:]=seq_idx
            
            temp = torch.zeros((4,2,16,16)) 
            for ii in range(0, len(idx)):
                temp[ii,:,:,:] = self.origin[i,:,idx[ii][0]:idx[ii][1],idx[ii][2]:idx[ii][3]] 
            tar = torch.zeros((4,2,16,16))
            for j in range(0, len(seq)):
            
                tar[j,:,:,:] = temp[seq[j],:,:,:] 
            for ii in range(0, len(idx)):
                s[i,:,idx[ii][0]:idx[ii][1],idx[ii][2]:idx[ii][3]] = tar[ii,:,:,:]
        return s,suff_idx
