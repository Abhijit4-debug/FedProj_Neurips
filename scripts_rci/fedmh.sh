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
    --nlp \
    --ntrials=2 \
    --data_ratios 0.1 \
    --rounds=15 \
    --num_users=10 \
    --frac=0.3 \
    --local_ep=1 \
    --local_bs=64 \
    --lr=0.001 \
    --distill_lr=0.00003 \
    --distill_E=1 \
    --distill_T=3 \
    --gamma=0.1 \
    --gamma2=0.0003 \
    --distill_data=mnli \
    --memory_bs=$bs \
    --ordering=None \
    --momentum=0.9 \
    --models=bert-tiny \
    --dataset=snli \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --datadir='/home/srip25/FedMH/.hf_data/' \
    --logdir='../neurips/' \
    --log_filename='resnet18_20E_1eKL_publicCifar100_Lambda_0.1_Memory_0.0003_100_dropout0.7_512'$bs \
    --alg='fedmh' \
    --iid_beta=0.5 \
    --niid_beta=0.3 \
    --seed=2023 \
    --gpu=0 \
    --print_freq=10
done 
