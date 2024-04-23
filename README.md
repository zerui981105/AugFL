# AugFL

AugFL is a Federated learning(FL) framework augmented by a large pretrained model(PM). We introduce a PM on server and transfer it's knowledge to improve FL training. 
The knowledge transfer is offloaded on the server based on a inexact variant of ADMM. Without sending the PM(or compressed PM) to clients, AugFL greatly reduces the communication, computation and storage costs. AugFL substantially outperforms Per-FedAvg and FedAvg, especially on sophisticated datasets. Specifically, AugFL achieves 4.5% over Per-FedAvg and 28.6% over FedAvg on Fashion-MNIST, 14.4% over Per-FedAvg and 47.4% over FedAvg on EMNIST, 11% over Per-FedAvg and 36.2% over FedAvg on CIFAR-10, and 18.3% over Per-FedAvg and 24.4% over FedAvg on CIFAR-100.

**<font size=6>Getting started</font>**:

```
git clone https://github.com/zerui981105/AugFL.git && cd AugFL
pip install -r requirements.txt
```

**<font size=6>Preparing data for clients</font>**:

Each client randomly samples two classes of data from the whole dataset.

```
python DataDivision.py
```

**<font size=6>Getting the PM</font>**:

You can either fetch PMs from a open source website,

```
python fetch_pretrained_teachers.sh
```

or pretrain a model by yourself

```
python train_teacher_model.py
```

**<font size=6>Training AugFL</font>**:

```
python launch_w_PM.py
```

**<font size=6>Experiment results</font>**:

We evaluate AugFL, FedAvg, Per-FedAvg on CIFAR-10, CIFAR-100, EMNIST, Fashion-MNIST.

| Dataset         | FedAvg          | Per-FedAvg      | AugFL(ours)     | 
| --------------- | --------------- | --------------- | --------------- | 
| CIFAR-10        | 54.41% ± 0.64%  | 67.17% ± 2.34%  | 74.49% ± 2.16%  | 
| CIFAR-100       | 46.18% ± 2.92%  | 48.25% ± 8.27%  | 57.35% ± 1.76%  | 
| EMNIST          | 54.34% ± 1.62%  | 69.74% ± 1.58%  | 79.89% ± 1.69%  |
| Fashion-MNIST   | 73.95% ± 3.68%  | 90.23% ± 3.23%  | 94.04% ± 1.16%  | 

Introducing the PM improves the performance like following:

![image](https://github.com/zerui981105/AugFL/blob/main/ablation.png)
