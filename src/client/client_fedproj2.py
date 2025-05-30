import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F

class Client_FedProj2(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, T, gamma, device, 
                 train_dl_local = None, test_dl_local = None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.T = T
        self.gamma = gamma
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.soft_targets = None
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True 
        
    def train(self, public_ds, is_proj=False, is_print=False):
        self.net.to(self.device)
        self.net.train()
        
        global_wavg = list(self.net.parameters())
        
        grad_dims = []
        for param in self.net.parameters():
            grad_dims.append(param.data.numel())
        grad_new = torch.Tensor(np.sum(grad_dims)).to(self.device)
        grad_glob = torch.Tensor(np.sum(grad_dims)).to(self.device)
        
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=256, shuffle=True, drop_last=False)

#         optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        
        break_flag = False
        epoch_loss = []
        for iteration in range(self.local_ep):
                
            batch_loss = []
            for batch_idx, (d1, d2) in enumerate(zip(self.ldr_train, public_dl)):
                                
                images, labels = d1
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                logits_d1 = self.net(images)
                
                teacher_x, teacher_y, teacher_logits = d2
                teacher_x = teacher_x.type(torch.LongTensor).float().to(self.device)
                teacher_logits = teacher_logits.type(torch.LongTensor).float().to(self.device)
                
                logits_student = self.net(teacher_x)
                
                if is_proj == False:
                    self.net.zero_grad()
                    loss = self.loss_func(logits_d1, labels)
                    loss.backward()
                else:
                    self.net.zero_grad()
                    loss = self.loss_func(logits_d1, labels)
                    loss.backward()
                    
                    grad_new.fill_(0.0)
                    count = 0
                    for param in self.net.parameters():
                        if param.grad is not None:
                            begin = 0 if count == 0 else sum(grad_dims[:count])
                            end = np.sum(grad_dims[:count + 1])
                            grad_new[begin: end].copy_(param.grad.data.view(-1))
                        count += 1
                    
                    self.net.zero_grad()
                    reg = 0.0
                    for param_index, param in enumerate(self.net.parameters()):
                        reg += torch.norm((param - global_wavg[param_index]))**2
                    
                    kd_loss = kl_criterion(F.log_softmax(logits_student/self.T, dim=1), F.softmax(teacher_logits/self.T, dim=1))

                    loss = self.T**2 * kd_loss + self.gamma * reg
                
                    loss.backward()
                    
                    grad_glob.fill_(0.0)
                    count = 0
                    for param in self.net.parameters():
                        if param.grad is not None:
                            begin = 0 if count == 0 else sum(grad_dims[:count])
                            end = np.sum(grad_dims[:count + 1])
                            grad_glob[begin: end].copy_(param.grad.data.view(-1))
                        count += 1
                    
                    dot_prod = torch.dot(grad_new, grad_glob)
                    if dot_prod.item() < 0:
                        corr = torch.dot(grad_new, grad_glob) / torch.dot(grad_glob, grad_glob)
                        grad_proj = grad_new - corr * grad_glob
                    else: 
                        grad_proj = grad_new
                        
                    count = 0
                    for param in self.net.parameters():
                        if param.grad is not None:
                            begin = 0 if count == 0 else sum(grad_dims[:count])
                            end = sum(grad_dims[:count + 1])
                            this_grad = grad_proj[begin: end].contiguous().view(param.grad.data.size())
                            param.grad.data.copy_(this_grad)
                        count += 1
                    
                optimizer.step()
                batch_loss.append(loss.item())
                
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
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=self.local_bs, shuffle=False, drop_last=False)
        
        outs = []
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
