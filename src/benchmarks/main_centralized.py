import sys
import os
import copy
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# Add path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# Import local modules
from src.data import get_dataset_global
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.resnet18(num_classes=100)

    def forward(self, x):
        return self.model(x)

class DatasetKD(Dataset):
    def __init__(self, dataset, soft_targets):
        self.dataset = dataset
        self.soft_targets = torch.tensor(soft_targets, dtype=torch.float32)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.soft_targets[index]

    def __len__(self):
        return len(self.dataset)

def eval_test(net, args, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return test_loss/len(testloader), acc

def knowledge_distillation(net_glob, public_ds, args):
    print("-" * 40)
    print("Starting Knowledge Distillation")
    print("-" * 40)

    # Get training data using imported get_dataset_global
    train_ds_global, test_ds_global, train_dl_global, \
    test_dl_global = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                      p_train=1.0, p_test=1.0)

    public_dl = DataLoader(public_ds, batch_size=128, shuffle=True)
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    
    net_glob.to(args.device)
    net_glob.train()
    global_wavg = [param.detach().clone() for param in net_glob.parameters()]

    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    T = 3  # Temperature
    gamma = 0.1  # Regularization parameter

    for epoch in range(5):
        total_kd_loss = 0.0
        num_batches = 0

        for batch_idx, (teacher_x, teacher_y, teacher_logits) in enumerate(public_dl):
            net_glob.zero_grad()

            teacher_x = teacher_x.to(args.device)
            teacher_logits = teacher_logits.to(args.device)

            logits_student = net_glob(teacher_x)

            kd_loss = kl_criterion(
                F.log_softmax(logits_student / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            )

            reg = torch.tensor(0.0, device=args.device)
            for param_index, param in enumerate(net_glob.parameters()):
                reg += torch.norm(param - global_wavg[param_index]) ** 2

            loss = T**2 * kd_loss + gamma * reg
            loss.backward()
            optimizer.step()

            total_kd_loss += kd_loss.item()
            num_batches += 1

        avg_kd_loss = total_kd_loss / num_batches
        print(f"Distill Epoch {epoch + 1} - Avg KL Loss: {avg_kd_loss:.6f}")

    print("-" * 40)
    print("KD Training Completed")
    print("-" * 40)
    
    _, acc_kd = eval_test(net_glob, args, test_dl_global)
    print("KD accuracy:", acc_kd)

def main_centralized(args):
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    
    print('-'*40)
    print('Getting Clients Data')
    train_ds_global, test_ds_global, train_dl_global, \
    test_dl_global = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                      p_train=args.p_train, p_test=args.p_test)

    print('-'*40)
    print('Building models')
    net_glob = Model().to(args.device)
    
    total_params = sum(p.numel() for p in net_glob.parameters())
    print(f'Total parameters: {total_params}')
    print('-'*40)

    start = time.time()
    
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    glob_acc = []
    loss_train = []

    for iteration in range(args.rounds):
        print(f'----- ROUND {iteration+1} -----')
        
        net_glob.train()
        batch_loss = []
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_dl_global):
            images, labels = images.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = net_glob(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        loss_avg = sum(batch_loss) / len(batch_loss)
        accuracy = 100. * correct / total
        
        print(f'-- Average Train loss: {loss_avg:.3f}')
        print(f'-- Training Accuracy: {accuracy:.2f}%')

        _, acc = eval_test(net_glob, args, test_dl_global)
        glob_acc.append(acc)
        
        print(f'-- Global Acc: {glob_acc[-1]:.3f}, Global Best Acc: {np.max(glob_acc):.3f}\n')
        loss_train.append(loss_avg)
        
        gc.collect()

    end = time.time()
    duration = end - start

    # Prepare public dataset for knowledge distillation
    public_train_ds, public_test_ds, _, _ = get_dataset_global(
        args.distill_dataset, args.datadir, batch_size=128,
        p_train=1.0, p_test=1.0
    )
    
    p_data = torch.utils.data.ConcatDataset([public_train_ds, public_test_ds])
    soft_t = np.random.randn(len(p_data), 100)  # Adjusted to match CIFAR-100 classes
    public_ds = DatasetKD(p_data, soft_t)
    
    # Perform knowledge distillation
    knowledge_distillation(net_glob, public_ds, args)

    print('-'*40)
    print('FINAL RESULTS')
    print(f'-- Global Acc Final: {glob_acc[-1]:.2f}')
    print(f'-- Global Best Acc: {np.max(glob_acc):.2f}')
    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)

    return glob_acc[-1], np.mean(glob_acc[-args.num_users:]), np.max(glob_acc), duration

def run_centralized(args, fname):
    alg_name = 'Centralized'
    
    exp_final_glob = []
    exp_avg_final_glob = []
    exp_best_glob = []
    exp_fl_time = []

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, f'Trial {trial+1}')

        final_glob, avg_final_glob, best_glob, duration = main_centralized(args)

        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_fl_time.append(duration/60)

        print('*'*40)
        print(' '*20, f'End of Trial {trial+1}')
        print(' '*20, 'Final Results')
        print(f'-- Global Final Acc: {exp_final_glob[-1]:.2f}')
        print(f'-- Global Avg Final Acc: {exp_avg_final_glob[-1]:.2f}')
        print(f'-- Global Best Acc: {exp_best_glob[-1]:.2f}')
        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, f'Avg {args.ntrials} Trial Results')
    print(f'-- Global Final Acc: {np.mean(exp_final_glob):.2f} +- {np.std(exp_final_glob):.2f}')
    print(f'-- Global Avg Final Acc: {np.mean(exp_avg_final_glob):.2f} +- {np.std(exp_avg_final_glob):.2f}')
    print(f'-- Global Best Acc: {np.mean(exp_best_glob):.2f} +- {np.std(exp_best_glob):.2f}')
    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, f'Avg {args.ntrials} Trial Results', file=text_file)
        print(f'-- Global Final Acc: {np.mean(exp_final_glob):.2f} +- {np.std(exp_final_glob):.2f}', file=text_file)
        print(f'-- Global Avg Final Acc: {np.mean(exp_avg_final_glob):.2f} +- {np.std(exp_avg_final_glob):.2f}', file=text_file)
        print(f'-- Global Best Acc: {np.mean(exp_best_glob):.2f} +- {np.std(exp_best_glob):.2f}', file=text_file)
        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)