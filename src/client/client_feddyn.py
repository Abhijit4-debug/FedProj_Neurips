import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Client_FedDyn(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local=None, test_dl_local=None, alpha=0.1):
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
        self.alpha = alpha
        self.beta = 0.1  # Linear penalty coefficient

        # Initialize prev_grads as a flat tensor on the correct device
        self.prev_grads = torch.zeros(
            sum(p.numel() for p in self.net.parameters()),
            device=self.device
        )

 

    def train(self, w_glob, is_print=False):
        self.net.to(self.device)
        self.net.train()
        self.net.load_state_dict(w_glob)

        # Move global weights to device
        w_glob_device = {k: v.to(self.device) for k, v in w_glob.items()}

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,weight_decay=0.00005)  
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.local_ep//2, gamma=0.1)

        epoch_loss = []

        for epoch in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long()

                optimizer.zero_grad()
                outputs = self.net(images)
                loss = self.loss_func(outputs, labels)

                # -------- Linear Penalty (FedDyn) -------- #
                curr_params = torch.cat([p.view(-1) for p in self.net.parameters()]).to(self.device)

                if hasattr(self, 'prev_grads') and self.prev_grads is not None:
                    self.prev_grads = self.prev_grads.to(self.device)
                    lin_penalty = torch.sum(curr_params * self.prev_grads)
                    loss -=  lin_penalty

                else:
                    lin_penalty = torch.tensor(0.0).to(self.device)

                # -------- Quadratic Penalty (safer version) -------- #
                quad_penalty = sum(
                    F.mse_loss(param, w_glob_device[name], reduction='mean')  # reduced mean instead of sum
                    for name, param in self.net.named_parameters()
                )
                loss += 0.01 / 2.0 * quad_penalty

                # Backpropagation
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)

                # Update prev_grads (FedDyn gradient storage)
                grads = torch.cat([param.grad.view(-1) for param in self.net.parameters()])
                self.prev_grads = grads.detach().clone().to(self.device)

                optimizer.step()
                batch_loss.append(loss.item())

                # print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | LinPen: {lin_penalty.item():.4f} | QuadPen: {quad_penalty.item():.4f}")
            scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

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
