# Benchmarking SOTA Federated Learning

This is a [PyTorch](https://pytorch.org/) implementation of the state-of-the-art **federated learning** benchmarks under **IID** ( Independent and Identically Distributed) and **Non-IID** data distributions.

## Algorithms

9 Global FL algorithms and 15 Personalized FL algorithms are supported in this code.

**Global** Algorithms:

| Algorithm | Publication |
| --------- | ----------- |
| FedAvg   | https://arxiv.org/abs/1602.05629 |
| FedProx  | https://arxiv.org/abs/1812.06127 |
| FedNova  | https://arxiv.org/abs/2007.07481 |
| Scaffold | https://arxiv.org/abs/1910.06378 |
| MOON     | https://arxiv.org/abs/2103.16257 |
| FedBN    | https://arxiv.org/abs/2102.07623 |
| FedDyn   | https://arxiv.org/abs/2111.04263 |
| FedDF    | https://arxiv.org/abs/2006.07242 |
| FedAvgM  | https://arxiv.org/abs/1909.06335 |
| FedavgM  | https://arxiv.org/abs/2002.06440 |

**Personalized** Algorithms:

| Algorithm | Publication |
| --------- | ----------- |
| Global + Fine-Tuning | |
| LG                   | https://arxiv.org/abs/2001.01523 |
| Per-FedAvg           | https://arxiv.org/abs/2002.07948 |
| CFL                  | https://arxiv.org/abs/1910.01991 |
| IFCA                 | https://arxiv.org/abs/2006.04088 |
| MTL                  | https://arxiv.org/abs/1705.10467 |
| Ditto                | https://arxiv.org/abs/2012.04221 |
| FedRep               | https://arxiv.org/abs/2102.07078 |
| FedPer               | https://arxiv.org/abs/1912.00818 |
| FedFOMO              | https://arxiv.org/abs/2012.08565 |
| pFedMe               | https://arxiv.org/abs/2006.08848 |
| APFL                 | https://arxiv.org/abs/2003.13461 |
| HeteroFL             | https://arxiv.org/abs/2010.01264 |
| SubFedAvg            | https://arxiv.org/abs/2105.00562 |
| FLIS                 | https://arxiv.org/abs/2112.07157 |


## Datasets

We support 6 widely used datasets: **CIFAR-10**, **CFIAR-100**, **FMNIST**, **PCIFAR10**, **PMNIST**, and **STL-10**.

TODO: how to align Datasets

## Usage

We provide scripts to run the algorithms, which are put under `scripts_rci/`. Here is an example to run the script:
```
cd scripts_rci
bash fedavg.sh
```

The descriptions of parameters are as follows:
| Parameter | Description |
| --------- | ----------- |
| ntrials      | The number of total runs. |
| rounds       | The number of communication rounds per run. |
| num_users    | The number of clients. |
| frac         | `TODO` |
| local_ep     | The number of local training epochs. |
| local_bs     | Local batch size. |
| lr           | The learning rate for local models. |
| momentum     | The momentum for the optimizer. |
| model        | Network architecture. Options: `TODO` |
| dataset      | The dataset for training and testing. Options are discussed above. |
| partition    | How datasets are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| datadir      | The path of datasets. |
| logdir       | The path to store logs. |
| log_filename | The folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3 folders named `1`, `2`, and `3`. |
| alg          | Federated learning algorithm. Options are discussed above. |
| beta         | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| local_view   | `TODO` |
| noise        | The maximum variance of Gaussian noise added to local party `TODO: what's loacl party?` |
| gpu          | The IDs of GPU to use. E.g., `TODO` |
| print_freq   | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |

## Reference
* Agem, [[Code](https://github.com/aimagelab/mammoth/blob/master/models/agem.py)]
* Gem, [[Code](https://github.com/aimagelab/mammoth/blob/32d0a806606eb8aecc0b2e8549478f7d44fd0f5e/models/gem.py#L36)]
* KD, [[Code](https://github.com/tyui592/knowledge_distillation)], [[Code](https://github.com/PkuDavidGuan/Distillation)], [[Code]()], [[Code]()]
