CASE_NAME="figurines"

gt_folder="../data/lerf_ovs/label"  # path to lerf_ovs/label

root_path="../"

python evaluate_iou_loc.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output/R-L/${CASE_NAME} \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/output/R-L/${CASE_NAME}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder}
