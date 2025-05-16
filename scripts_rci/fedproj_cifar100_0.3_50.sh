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

for bs in 12500
do
#     dir='../results_fedkd/fedavg/cifar10'
#     if [ ! -e $dir ]; then
#     mkdir -p $dir
#     fi
    
    python ../main.py \
    --ntrials=3 \
    --rounds=100 \
    --num_users=50 \
    --frac=0.1 \
    --local_ep=20 \
    --local_bs=64 \
    --lr=0.001 \
    --distill_lr=0.00001 \
    --distill_E=3 \
    --distill_T=3 \
    --gamma=0.01 \
    --gamma2=0.0 \
    --distill_data=tinyimagenet \
    --memory_bs=$bs \
    --ordering=None \
    --momentum=0.9 \
    --model=resnet18 \
    --dataset=cifar100 \
    --p_train=1.0 \
    --partition='niid-labeldir' \
    --datadir='../../data/' \
    --logdir='../new_results/absolute_new/step_reduction' \
    --log_filename='resnet18_distill_E=3_125000_wd_yes_client_l2_0.0gamma2_0.0_dropout_0.7_fedproj_client_seperateloaders_batchsize_768_off90_100'$bs \
    --alg='fedproj' \
    --iid_beta=0.5 \
    --niid_beta=0.1 \
    --seed=2023 \
    --gpu=0 \
    --print_freq=10
done 
