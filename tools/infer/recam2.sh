# cd /m2v_intern/mengzijie/m2v_camclone_v2/
cd /ytech_m2v4_hdd/mengzijie/m2v_camclone_v2/

# [新增] 提前创建 log 文件夹，防止因为目录不存在而报错
mkdir -p log

# [新增] 用 { 把后续所有命令包起来
{
    CKPT_DIR="/ytech_m2v4_hdd/mengzijie/m2v_camclone_v2/exps/0016--1b_camclonemaster_node_12/checkpoints"

    SUBDIR_LIST=("checkpoint-1272000")  
    echo "start envir"

    source /ytech_m2v4_hdd/mengzijie/m2v0524/bin/activate
    echo "envir is ok"
    # export CUDA_VISIBLE_DEVICES=7

    cfg=7.5
    for SUBDIR_NAME in "${SUBDIR_LIST[@]}"; do
        SUBDIR="$CKPT_DIR/$SUBDIR_NAME"
        if [ -d "$SUBDIR" ]; then
            TRANSFORMER_CKPT_PATH="$SUBDIR/ema/transformer.ckpt"
            echo "ckpt is $TRANSFORMER_CKPT_PATH"
            TEST_DIR="test_dir/demo_recam1"
            echo "Test directory is $TEST_DIR"
            
            # 注意：这里的 \ 续行符后一定不能有空格
            bash scripts/dist_run.sh \
                python scripts/m2v_dist_infer_i2v_recam.py \
                /ytech_m2v4_hdd/mengzijie/m2v_camclone_v2/exps/0016--1b_camclonemaster_node_12/config.yml \
                --data.path /ytech_m2v4_hdd/mengzijie/m2v_camclone_v2/160.csv \
                --data.id_column id \
                --data.index_column index \
                --data.t5_prompt_embed_column None \
                --data.caption_column prompt \
                --data.ref_path_column ref_videos_old \
                --data.video_path_column ref_videos2_old \
                --data.content_ref_path_column ref_videos2_old \
                --data.num_samples 160 \
                --data.batch_size 1 \
                --data.cache_dir None \
                --data.crop_type None \
                --test_dir "$TEST_DIR" \
                --transformer_ckpt_path "$TRANSFORMER_CKPT_PATH" \
                --negative_prompt "animation, 2d animation, 3d animation, Anime, Cartoon, blurry, deformed, disfigured, low quality, text, collage, grainy, logo, no visual content, blurred effect, striped background, abstract, illustration, computer generated, distorted" \
                --width 672 \
                --height 384 \
                --fps 15 \
                --num_frames 41 \
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

# [新增] 括号结束。2>&1 表示把报错也算作输出，tee 会把输出同时发给终端和 log/log.txt
} 2>&1 | tee log/log.txt