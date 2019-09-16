export OMP_NUM_THREADS=2
export NGPUS=4
export CKPOINT="model_final.pth"
python -m torch.distributed.launch --nproc_per_node=$NGPUS ../tools/test_net.py --config-file "e2e_faster_rcnn_R_50_FPN_1x.yaml"  MODEL.WEIGHT $CKPOINT MODEL.ROI_HEADS.SCORE_THRESH 0.0001 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 500 TEST.IMS_PER_BATCH 8 DATASETS.TEST "(\"openimagesv4_challenge_val\",)"
python -m torch.distributed.launch --nproc_per_node=$NGPUS ../tools/test_net.py --config-file "e2e_faster_rcnn_R_50_FPN_1x.yaml"  MODEL.WEIGHT $CKPOINT MODEL.ROI_HEADS.SCORE_THRESH 0.0001 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 500 TEST.IMS_PER_BATCH 8 DATASETS.TEST "(\"openimagesv4_novel\",)"
python -m torch.distributed.launch --nproc_per_node=$NGPUS ../tools/test_net.py --config-file "e2e_faster_rcnn_R_50_FPN_1x.yaml"  MODEL.WEIGHT $CKPOINT MODEL.ROI_HEADS.SCORE_THRESH 0.0001 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 100 TEST.IMS_PER_BATCH 8 DATASETS.TEST "(\"visualgenome\",)"