CASE_NAME="Dimsum"

gt_folder="../data/RGBT-Scenes/label"

root_path="../"

python evaluate_rgbt.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output/R-T-L/${CASE_NAME}_2 \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/output/R-T-L/${CASE_NAME}_2/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder}
