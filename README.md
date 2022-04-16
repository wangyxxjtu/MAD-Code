# Make Manual Architecture Design Great Again

This is is a non-official PyTorch implementation of this work (2022):

[Make Manual Architecture Design Great Again](https://arxiv.org/pdf/2108.08607.pdf)

## Introduction
This is an exploration work, the authors made a comprehensive study on the hyperparameters and Arch. configurations in the manual-designed networks,
and found the manual-designed networks could even surpass the NAS-based method if we could appropriately use the configurations: 

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/flops_acc.png" width="745" alt="workflow" />

The final presented network could surpass the many sota methods including NAS-based ones on ImageNet, this work may provide some guidance for the manual network design:

<img src="https://github.com/wangyxxjtu/MAD-Code/blob/master/result.png" width="845" alt="workflow" />

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
If you use our code, please cite our work:
``` bash
{@article{wangyx_pcnet,
  author    = {Yaxiong Wang and
               Yunchao Wei and
               Xueming Qian and
               Li Zhu and
               Yi Yang},
  title     = {Generating Superpixels for High-resolution Images with Decoupled Patch
               Calibration},
  journal   = {CoRR},
  volume    = {abs/2108.08607},
  year      = {2021},
  url       = {https://arxiv.org/abs/2108.08607},
  eprinttype = {arXiv},
  eprint    = {2108.08607}}
```
