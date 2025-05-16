import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader



import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


# class CLLoss(nn.Module):
#     def __init__(self, temperature=0.05, topk_neg=2, divergence_weight=0):
#         super().__init__()
#         self.temperature = temperature
#         self.topk_neg = topk_neg
#         self.divergence_weight = divergence_weight

#     def forward(self, old_feat, new_feat, target=None, reduction='mean'):
#         old_feat = F.normalize(old_feat, dim=1)
#         new_feat = F.normalize(new_feat, dim=1)

#         pos_sim = torch.sum(old_feat * new_feat, dim=1, keepdim=True)

#         neg_sim = torch.matmul(new_feat, old_feat.t())  # (B,B)
#         mask = ~torch.eye(neg_sim.size(0), dtype=torch.bool, device=neg_sim.device)
#         neg_sim = neg_sim[mask].view(neg_sim.size(0), -1)

#         neg_sim_topk, _ = torch.topk(neg_sim, min(self.topk_neg, neg_sim.size(1)), dim=1)
#         logits = torch.cat([pos_sim, neg_sim_topk], dim=1) / self.temperature
#         labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

#         cl_loss = F.cross_entropy(logits, labels, reduction=reduction)

#         if self.divergence_weight > 0:
#             div_loss = F.mse_loss(old_feat, new_feat, reduction=reduction)
#             cl_loss += self.divergence_weight * div_loss

#         return cl_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import copy


class BatchedCLLoss(nn.Module):
    def __init__(self, temperature=0.05, divergence_weight=0.1, lambda_threshold=0.8):
        super().__init__()
        self.temperature = temperature
        self.divergence_weight = divergence_weight
        self.lambda_threshold = lambda_threshold

    def forward(self, old_feat, new_feat, target, reduction='mean'):
        if target is None:
            raise ValueError("FedRCL requires target labels for contrastive loss")

        old_feat = F.normalize(old_feat, dim=1)
        new_feat = F.normalize(new_feat, dim=1)

        # Similarity matrix
        sim_matrix = torch.matmul(new_feat, old_feat.T) / self.temperature  # [B, B]

        B = target.size(0)
        labels = target
        class_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        diag_mask = ~torch.eye(B, dtype=torch.bool, device=target.device)
        mask = class_mask & diag_mask  # [B, B], mask out self

        sim_matrix = sim_matrix.masked_fill(~mask, float('-inf'))
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Average over positive samples
        mean_log_prob_pos = (log_prob * mask).sum(1) / mask.sum(1).clamp(min=1)
        cl_loss = -mean_log_prob_pos.mean()

        # Optional divergence loss
        div_loss = F.mse_loss(new_feat, old_feat, reduction=reduction)
        cl_loss += self.divergence_weight * div_loss

        return cl_loss


class Client_RCL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local=None, test_dl_local=None, mu=0.001, temperature=0.5,
                 loss_weights=None, use_mixed_precision=False, print_debug=False):
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
        self.mu = mu
        self.temperature = temperature

        self.prevnet = copy.deepcopy(self.net)
        self.use_mixed_precision = use_mixed_precision
        self.print_debug = print_debug
        self.scaler = GradScaler() if self.use_mixed_precision else None

        self.cl_loss_fn = BatchedCLLoss(temperature=0.05,
                                        divergence_weight=0.1,
                                        lambda_threshold=0.8)

        self.loss_weights = loss_weights or {
            'cls': 1.0,
            'scl': 0.1,
            'prox': 0.01,
            'cossim': 0.0,
        }

    def _forward_pass(self, images):
        local_out = self.net(images, no_relu=True)
        with torch.no_grad():
            global_out = self.prevnet(images, no_relu=True)
        return local_out, global_out

    def _compute_losses(self, local_out, global_out, labels):
        losses = defaultdict(float)

        logits = local_out["logit"]
        losses["cls"] = F.cross_entropy(logits, labels)

        # Proximal loss
        prox_loss = 0.0
        fixed_params = {n: p for n, p in self.prevnet.named_parameters()}
        for n, p in self.net.named_parameters():
            prox_loss += ((p - fixed_params[n].detach()) ** 2).sum()
        losses["prox"] = prox_loss

        losses["cossim"] = torch.tensor(0.0, device=self.device)  

        # Use only last layer for contrastive loss
        last_layer_key = sorted([k for k in local_out if k.startswith('layer')])[-1]
        local_feat = local_out[last_layer_key]
        global_feat = global_out[last_layer_key]

        if local_feat.ndim == 4:
            local_feat = F.adaptive_avg_pool2d(local_feat, 1).view(local_feat.size(0), -1)
            global_feat = F.adaptive_avg_pool2d(global_feat, 1).view(global_feat.size(0), -1)

        if labels.size(0) >= 4:
            losses["scl"] = self.cl_loss_fn(global_feat, local_feat, labels)
        else:
            losses["scl"] = torch.tensor(0.0, device=self.device)

        return losses

    def train(self, is_print=False):
        self.net.to(self.device).train()
        self.prevnet.to(self.device).eval()

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.local_ep // 2), gamma=0.1)

        epoch_loss = []

        for epoch in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                if self.use_mixed_precision:
                    with autocast():
                        local_out, global_out = self._forward_pass(images)
                        losses = self._compute_losses(local_out, global_out, labels)
                        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    local_out, global_out = self._forward_pass(images)
                    losses = self._compute_losses(local_out, global_out, labels)
                    total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                    optimizer.step()

                batch_loss.append(total_loss.item())

                # if batch_idx % 10 == 0:
                #     print(f'Client {self.name} | Epoch {epoch + 1}/{self.local_ep} | Batch {batch_idx} | Loss: {total_loss.item():.4f}')

            scheduler.step()
            avg_epoch_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_epoch_loss)

            print(f'Client {self.name} | Epoch {epoch + 1}/{self.local_ep} | Avg Loss: {avg_epoch_loss:.4f}')

        self.prevnet = copy.deepcopy(self.net).cpu().eval()

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

                _,_,output = self.net(data)
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

                _,_,output = self.net(data)
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

                _,_,output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
