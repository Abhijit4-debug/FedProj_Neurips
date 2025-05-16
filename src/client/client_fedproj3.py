import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class Client_FedProj3(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, T, gamma, gamma2, device, 
                 train_dl_local = None, test_dl_local = None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.T = T
        self.gamma = gamma
        self.gamma2 = gamma2
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.soft_targets = None
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True
        
        self.gamma=0


    # def train(self, public_ds, is_proj=False, is_print=False):
    #     self.net.to(self.device)
    #     self.net.train()
        
    #     #optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
    #     optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,weight_decay=0.00005)  
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.local_ep//2, gamma=0.1)
    #     epoch_loss = []
    #     for iteration in range(self.local_ep):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             labels = labels.type(torch.LongTensor).to(self.device)
                
    #             self.net.zero_grad()
    #             optimizer.zero_grad()
    #             # log_probs = self.net(images)
    #             logits = self.net(images)
    #             if isinstance(logits, tuple):
    #                 logits = logits[0]
    #             loss = self.loss_func(logits, labels)
    #             # loss = self.loss_func(log_probs, labels)
    #             loss.backward() 
                        
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         scheduler.step()
                
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    #     return sum(epoch_loss) / len(epoch_loss)
        
    def train(self, public_ds, is_proj=False, is_print=False):
            self.net.to(self.device)
            self.net.train()
            global_wavg = list(self.net.parameters())
            grad_dims = []
            for param in self.net.parameters():
                grad_dims.append(param.data.numel())
            grad_new = torch.Tensor(np.sum(grad_dims)).to(self.device)
            grad_glob = torch.Tensor(np.sum(grad_dims)).to(self.device)
            grad_proj = torch.Tensor(np.sum(grad_dims)).to(self.device)  # Initialize grad_proj
            grad_proj = torch.zeros(np.sum(grad_dims), device=self.device)

            
            public_dl = torch.utils.data.DataLoader(public_ds, batch_size=512, shuffle=True, drop_last=False)
            public_iter = iter(public_dl)
            # print(f"private data loader {len(self.ldr_train)}")
            # print(f"public data loader {len(public_dl)}")


    #         optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,weight_decay=0.00005)  
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.local_ep//2, gamma=0.1)
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
        
            break_flag = False
            epoch_loss = []
            for iteration in range(self.local_ep):
                    
                batch_loss = []
                # for batch_idx, (d1, d2) in enumerate(zip(self.ldr_train, public_dl)):
                for batch_idx, (images, labels) in enumerate(self.ldr_train):  # No public_dl
                                    
                    # images, labels = d1
                    images, labels =images.to(self.device), labels.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)
                    logits_d1 = self.net(images)
                    if isinstance(logits_d1, tuple):
                        logits_d1 = logits_d1[0]
                    # logits_d1 = self.net(images)

                    if is_proj == False:
                        self.net.zero_grad()
                        optimizer.zero_grad()
                        loss = self.loss_func(logits_d1, labels)
                        loss.backward()
                    else:

                        try:
                            teacher_x, teacher_y, teacher_logits, teacher_features = next(public_iter)
                        except StopIteration:
                            public_iter = iter(public_dl)  # Reset the iterator
                            teacher_x, teacher_y, teacher_logits, teacher_features = next(public_iter)  # Get new batch

                        #  teacher_x, teacher_y, teacher_logits, teacher_features = d2
                        teacher_x = teacher_x.type(torch.LongTensor).float().to(self.device)
                        teacher_logits = teacher_logits.type(torch.LongTensor).float().to(self.device)
                        teacher_features = teacher_features.to(self.device)
                        logits_student, features_student = self.net(teacher_x)
                      

                        self.net.zero_grad()
                        optimizer.zero_grad()
                        loss = self.loss_func(logits_d1, labels)
                        loss.backward()
                        grad_proj.fill_(0.0)
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
                        feature_loss = feature_loss_function(features_student, teacher_features.detach()) 
                    
                        loss = self.T**2 * kd_loss + self.gamma * reg + self.gamma2 * feature_loss
                    
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
                        grad_glob_norm_sq = torch.dot(grad_glob, grad_glob) 
                        if dot_prod.item() < 0 and grad_glob_norm_sq > 1e-6:
                            # Use a numerically stable version with clipping
                            corr = dot_prod / (grad_glob_norm_sq + 1e-8)  # Add small epsilon for stability
                            corr = torch.clamp(corr, -10.0, 10.0)  # Prevent extreme values
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
        features = []
        cnt=0
        with torch.no_grad():
            for data, _,_,_ in public_dl:
                data = data.to(self.device)
                out, feat = self.net(data)
                outs.append(out.detach().cpu())
                features.append(feat.detach().cpu())
#             if cnt==0:
#                 print(f'Features shape: {features[-1].shape}')
#             cnt+=1

        outputs = torch.cat(outs).numpy()
        features_t = torch.cat(features).numpy()
        return outputs, features_t
    
    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                
                output, _ = self.net(data)
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
                
                output, _ = self.net(data)
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
                
                output, _ = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
