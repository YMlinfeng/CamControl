cd /m2v_intern2/luoyawen/m2v_camclone_v2/
CKPT_DIR="/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0016--1b_camclonemaster_node_12/checkpoints"

SUBDIR_LIST=("checkpoint-1272000")  # 替换为实际的子文件夹名称

source /m2v_intern/luoyawen/Miniconda/miniconda/bin/activate m2v0524

cfg=7.5
for SUBDIR_NAME in "${SUBDIR_LIST[@]}"; do
    SUBDIR="$CKPT_DIR/$SUBDIR_NAME"
    if [ -d "$SUBDIR" ]; then
        # 替换命令中的地址
        TRANSFORMER_CKPT_PATH="$SUBDIR/ema/transformer.ckpt"
        echo "ckpt is $TRANSFORMER_CKPT_PATH"
        TEST_DIR="test_dir/demo_recam"
        echo "Test directory is $TEST_DIR"

        bash scripts/dist_run.sh \
            python scripts/m2v_dist_infer_i2v_recam.py \
            /m2v_intern2/luoyawen/m2v_camclone_v2/exps/0016--1b_camclonemaster_node_12/config.yml \
            --data.path /m2v_intern/public_datasets/Camera_Dataset/Testset/Csv/testset/testset_3_arc_left.csv \
            --data.t5_prompt_embed_column None \
            --data.caption_column caption \
            --data.ref_path_column ref_video_path \
            --data.video_path_column video_path \
            --data.content_ref_path_column video_path \
            --data.num_samples 24 \
            --data.batch_size 1 \
            --data.cache_dir None \
            --data.crop_type None \
            --test_dir "$TEST_DIR" \
            --transformer_ckpt_path "$TRANSFORMER_CKPT_PATH" \
            --negative_prompt "animation, 2d animation, 3d animation, Anime, Cartoon, blurry, deformed, disfigured, low quality, text, collage, grainy, logo, no visual content, blurred effect, striped background, abstract, illustration, computer generated, distorted" \
            --width 672 \
            --height 384 \
            --fps 15 \
            --num_frames 77 \
            --guidance_scale $cfg \
            --seed 42 \
            --num_inference_steps 50 \
            --timestep_shift 10.0 \

        if [ $? -ne 0 ]; then
            echo "Script execution failed for $SUBDIR_NAME"
        else
            echo "Script executed successfully for $SUBDIR_NAME"
        fi
    else
        echo "Directory $SUBDIR does not exist."
    fi
done
