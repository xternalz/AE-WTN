export NGPUS=4
OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=$NGPUS ../tools/train_net.py --config-file "e2e_faster_rcnn_R_50_FPN_1x.yaml"