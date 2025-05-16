#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=fedavg_cifar10_iid
#SBATCH --err=results/fedavg_cifar10_iid.err
#SBATCH --out=results/fedavg_cifar10_iid.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for lr in 0.001
do
#     dir='../results_fedkd/fedavg/cifar10'
#     if [ ! -e $dir ]; then
#     mkdir -p $dir
#     fi
    
    python ../main.py \
    --ntrials=3 \
    --rounds=60 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=20 \
    --local_bs=64 \
    --lr=$lr \
    --momentum=0.9 \
    --model=resnet18 \
    --dataset=cinic10 \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --datadir='../../data/' \
    --logdir='../neurips/' \
    --log_filename='resnet18_clients50_C20_E10_lr'$lr'_adam_0.3_100' \
    --alg='moon' \
    --iid_beta=0.5 \
    --niid_beta=0.3 \
    --gpu=0 \
    --print_freq=10
done 
