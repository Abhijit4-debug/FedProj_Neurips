import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F

class Client_FedDF(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, 
                 train_dl_local = None, test_dl_local = None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True 
        
    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,weight_decay=0.00005)  
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.local_ep//2, gamma=0.1)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward() 
                        
                optimizer.step()
                batch_loss.append(loss.item())
            scheduler.step()
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
#         if self.save_best: 
#             _, acc = self.eval_test()
#             if acc > self.acc_best:
#                 self.acc_best = acc 
        
        return sum(epoch_loss) / len(epoch_loss)
    
    def get_state_dict(self):
        return self.net.state_dict()
    def get_best_acc(self):
        return self.acc_best
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    
    def inference(self, public_ds):
        self.net.to(self.device)
        self.net.eval()
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=self.local_bs, shuffle=False, drop_last=False)
        
        outs = []
        with torch.no_grad(): 
            for data, _,_ in public_dl:
                data = data.to(self.device)
                out = self.net(data)
                outs.append(out.detach().cpu())

        outputs = torch.cat(outs).numpy()
        return outputs
    
    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy
    
    def eval_test_perclass(self, glob_dl, nclass=10):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        acc_class = [0 for c in range(nclass)]
        class_cnt = [0 for c in range(nclass)]
        
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                for c in range(nclass):
                    acc_class[c] = acc_class[c] \
                    + ((pred == target) * (target == c)).float().long().sum().cpu() #(max(labels == c).sum(), 1)
                    
                    class_cnt[c] = class_cnt[c] + (target == c).sum().long().cpu()
        
        test_loss /= len(glob_dl.dataset)
        for c in range(nclass):
            acc_class[c] = acc_class[c]/class_cnt[c]
        
        return test_loss, acc_class
    
    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy