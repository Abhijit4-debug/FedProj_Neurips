import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast

class Client_FedProj(object):
    

    def __init__(self, name, model, local_bs, local_ep, optim, lr, momentum, local_wd, scheduler, device, 
                train_dl_local = None, test_dl_local = None, nlp=False):
        self.mixed_precision_training = False
        self.name = name 
        self.net = model.cpu()
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.optim = optim
        self.lr = lr
        self.momentum = momentum 
        self.local_wd = local_wd
        self.scheduler = scheduler
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True 
        self.nlp = nlp
        
    def train(self, public_ds, is_proj=False):
        self.net.to(self.device)
        self.net.train()
        
        # Optimizer
        if self.optim == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.local_wd)
        elif self.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.local_wd)
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)

        # Scheduler
        if self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.local_ep, eta_min=self.lr / 100)
        elif self.scheduler == "cosineW":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.local_ep // 2, T_mult=1, eta_min=self.lr / 100)

        if self.mixed_precision_training:
            scaler = GradScaler()

        # Projection Gradients (used for FedPAC, FedKD, etc.)
        if is_proj and public_ds is not None:
            grad_shapes = [param.shape for param in self.net.parameters()]
            grad_sizes = [param.numel() for param in self.net.parameters()]
            grad_total = sum(grad_sizes)
            grad_new = torch.zeros(grad_total).to(self.device)
            grad_glob = torch.zeros(grad_total).to(self.device)
            grad_proj = torch.zeros(grad_total).to(self.device)
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            public_dl = torch.utils.data.DataLoader(public_ds, batch_size=128, shuffle=True, drop_last=False)
            public_iter = iter(public_dl)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, data in enumerate(self.ldr_train):
                if data is None:
                    continue

                # NLP or Vision Input
                if self.nlp:
                    data_in = {k: v.to(self.device) for k, v in data.items()}
                    labels = data["labels"].to(self.device)
                else:
                    data_in, labels = data
                    data_in, labels = data_in.to(self.device), labels.to(self.device)
                    labels = labels.type(torch.LongTensor).to(self.device)

                self.net.zero_grad()
                optimizer.zero_grad()

                if is_proj and public_ds is not None:
                    # Task loss
                    if self.nlp:
                        outputs = self.net(**data_in)
                        loss = getattr(outputs, "loss", None)
                        if loss is None:
                            loss = self.loss_func(outputs.logits, labels)
                    else:
                        log_probs = self.net(data_in)
                        loss = self.loss_func(log_probs, labels)
                    loss.backward()

                    # Store gradients
                    grad_new.fill_(0.0)
                    offset = 0
                    for param, size in zip(self.net.parameters(), grad_sizes):
                        if param.grad is not None:
                            grad_new[offset:offset + size].copy_(param.grad.view(-1))
                        offset += size

                    # Public batch for projection
                    try:
                        teacher_x, _, teacher_logits = next(public_iter)
                    except StopIteration:
                        public_iter = iter(public_dl)
                        teacher_x, _, teacher_logits = next(public_iter)

                    if self.nlp:
                        teacher_x = {k: v.to(self.device) for k, v in teacher_x.items()}
                    else:
                        teacher_x = teacher_x.float().to(self.device)
                    teacher_logits = teacher_logits.float().to(self.device)

                    self.net.zero_grad()
                    if self.nlp:
                        student_logits = self.net(**teacher_x).logits
                    else:
                        student_logits = self.net(teacher_x)

                    kd_loss = kl_criterion(
                        F.log_softmax(student_logits / 3, dim=1),
                        F.softmax(teacher_logits / 3, dim=1)
                    )
                    kd_loss.backward()

                    grad_glob.fill_(0.0)
                    offset = 0
                    for param, size in zip(self.net.parameters(), grad_sizes):
                        if param.grad is not None:
                            grad_glob[offset:offset + size].copy_(param.grad.view(-1))
                        offset += size

                    dot_prod = torch.dot(grad_new, grad_glob)
                    grad_glob_norm_sq = torch.dot(grad_glob, grad_glob)

                    if dot_prod.item() < 0 and grad_glob_norm_sq > 1e-6:
                        corr = dot_prod / (grad_glob_norm_sq + 1e-8)
                        corr = torch.clamp(corr, -10.0, 10.0)
                        grad_proj = grad_new - corr * grad_glob
                    else:
                        grad_proj = grad_new

                    # Apply projected gradient
                    offset = 0
                    for param, size, shape in zip(self.net.parameters(), grad_sizes, grad_shapes):
                        if param.grad is not None:
                            this_grad = grad_proj[offset:offset + size].contiguous().view(shape)
                            param.grad.copy_(this_grad)
                        offset += size

                    optimizer.step()
                    batch_loss.append(loss.item())
                    continue  # skip scaler/scheduler, handled already

                # Regular training path
                if self.mixed_precision_training:
                    with autocast():
                        if self.nlp:
                            outputs = self.net(**data_in)
                            loss = getattr(outputs, "loss", None)
                            if loss is None:
                                loss = self.loss_func(outputs.logits, labels)
                        else:
                            log_probs = self.net(data_in)
                            loss = self.loss_func(log_probs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if self.nlp:
                        outputs = self.net(**data_in)
                        loss = getattr(outputs, "loss", None)
                        if loss is None:
                            loss = self.loss_func(outputs.logits, labels)
                    else:
                        log_probs = self.net(data_in)
                        loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                batch_loss.append(loss.item())

            if self.scheduler != "none":
                scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.net.cpu()
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
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=64, shuffle=False, drop_last=False)
        self.net.eval()
        self.net.to(self.device)
        
        outs = []
        with torch.no_grad(): 
            for data, *_ in public_dl:
                if self.nlp: 
                    data = {k:v.to(self.device) for k,v in data.items() if k != 'labels'}
                    out = self.net(**data)
                    outs.append(out.logits)
                else: 
                    data = data.to(self.device)
                    out = self.net(data)
                    outs.append(out.detach().cpu())

        self.net.cpu()
        outputs = torch.cat(outs)
            #print(f'out {out.shape}, output: {outputs.shape}')
        return outputs
    
    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        correct = 0
        with torch.no_grad():
            for data in self.ldr_test:
                if self.nlp: 
                    data = {k:v.to(self.device) for k,v in data.items()}
                    out = self.net(**data)
                    output = out.logits
                    predictions = torch.argmax(output, dim=-1)
                    correct += (predictions == data['labels']).sum()
                else: 
                    data, target = data
                    data, target = data.to(self.device), target.to(self.device)
                    out = self.net(data)
                    output = out.detach().cpu()
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        accuracy = 100. * correct.item() / len(self.ldr_test.dataset)
        return accuracy
    
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
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        accuracy = 100. * correct / len(glob_dl.dataset)
        return accuracy
    
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