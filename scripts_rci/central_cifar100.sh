#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=feddf_cifar10_iid
#SBATCH --err=results/feddf_cifar10_iid.err
#SBATCH --out=results/feddf_cifar10_iid.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for bs in 10000
do
#     dir='../results_fedkd/fedavg/cifar10'
#     if [ ! -e $dir ]; then
#     mkdir -p $dir
#     fi
    
    python ../main.py \
    --ntrials=1 \
    --rounds=100 \
    --num_users=50 \
    --frac=0.1 \
    --local_ep=20 \
    --local_bs=64 \
    --lr=0.001 \
    --distill_lr=0.00001 \
    --distill_E=1 \
    --distill_T=3 \
    --gamma=0.3 \
    --gamma2=0.0001 \
    --distill_data=cifar100 \
    --memory_bs=$bs \
    --ordering=None \
    --momentum=0.9 \
    --model=resnet8 \
    --dataset=cinic10 \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --datadir='../../data/' \
    --logdir='../new_results//' \
    --log_filename='resnet8_20E_1eKL_publicCifar100_Lambda_0.0_Gamma2_0.0001_Memory_0.5_100'$bs \
    --alg='centralized' \
    --iid_beta=0.5 \
    --niid_beta=0.5 \
    --seed=2023 \
    --gpu=0 \
    --print_freq=10
done 
