# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u -m torch.distributed.launch --nproc_per_node 8 train_dist.py &
OMP_NUM_THREADS=2 nohup python -u -m torch.distributed.launch --nproc_per_node 8 train_dist_ema.py &
