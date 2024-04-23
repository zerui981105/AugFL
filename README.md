# AugFL

AugFL is a Federated learning(FL) framework augmented by a large pretrained model(PM). We introduce a PM on server and transfer it's knowledge to improve FL training. 
The knowledge transfer is offloaded on the server based on a inexact variant of ADMM. Without sending the PM(or compressed PM) to clients, AugFL greatly reduces the communication, computation and storage costs.
