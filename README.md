# Narrowing the Gap Between Manual Architecture Design and Network Architecture Search

This is is a PyTorch implementation of our work (2023):

[Narrowing the Gap Between Manual Architecture Design and Network Architecture Search]

## Introduction
This work is an interesting exploration, the authors made a comprehensive study on the hyperparameters and Arch. configurations in the manual-designed networks, and found the manual-designed networks could even surpass the NAS-based method if we could appropriately use the configurations: 

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/figures/fig11.jpg" width="855" alt="workflow" />

The final presented network could surpass the many sota methods including NAS-based ones on ImageNet, this work may provide some guidance for the manual network design:

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/figures/table1.png" width="845" alt="workflow" />

Our model also attain better performance than many SOTA networks on object detection and segmentation:

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/figures/table2.png" width="845" alt="workflow" />

Visualization of detection and segmentation:

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/figures/fig3.png" width="845" alt="workflow" />

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/figures/fig4.png" width="845" alt="workflow" />

## Training and Testing
run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -u -m torch.distributed.launch --nproc_per_node 4 train_dist_ema.py
```
or simply run:
```
sh run.sh
```

## Citation

Waiting to be published.
