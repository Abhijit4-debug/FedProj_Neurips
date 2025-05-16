#!/bin/bash

alpha="0.5"

self_reg="0"
task_weights="[[0.1, 0.1, 0.8]]"
tune="no_tuning"
n_candidates="10"

private_ds="marc"
public_ds="yelp"

alg="fedet"
gpu="0"

logdir=".neurips/results_nlp_pri-${private_ds}_pub-${public_ds}_"
log_filename="$alg"

python ./main.py \
--nlp \
--tune_lambda "$tune" \
--n_candidates "$n_candidates" \
--gpu "$gpu" \
--num_threads -1 \
--train_size 100000 \
--public_size 30000 \
--task_weights "$task_weights" \
--self_reg ${self_reg} \
--ntrials 3 \
--rounds 15 \
--nclusters 1 \
--num_users 10 \
--fracs 0.3 \
--data_ratios 1.0 \
--models bert-tiny \
--local_ep 1 \
--local_bs 64 \
--optim adam \
--lr 3e-5 \
--lr_scheduler none \
--local_wd 0 \
--dataset "$private_ds" \
--distill_dataset "$public_ds" \
--distill_lr 3e-5 \
--distill_wd 0 \
--distill_E 1 \
--distill_T 3 \
--partition niid-labeldir \
--datadir /home/srip25/FedMH/.hf_data/ \
--logdir "$logdir" \
--log_filename "$log_filename" \
--alg "$alg" \
--niid_beta "$alpha" \
--seed 2023
