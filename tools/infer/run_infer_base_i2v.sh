cd /m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2
bash scripts/dist_run.sh \
        python scripts/m2v_dist_infer.py \
        /m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.yml \
    --data.path /m2v_intern/luoyawen/Dataset/Good_Camera/Csv/0403_video_paths_testset_updated_captioned_with_frames.csv \
    --data.caption_column caption \
    --data.condition_image_path_column first_frame_path \
    --data.num_samples 80 \
    --data.batch_size 1 \
    --data.cache_dir None \
    --data.crop_type None \
    --test_dir testdir/test_results_i2v \
    --transformer_ckpt_path /m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt \
    --negative_prompt "animation, 2d animation, 3d animation, Anime, Cartoon, blurry, deformed, disfigured, low quality, software, text, signature, collage, grainy, logo, no visual content, blurred effect, striped background, abstract, illustration, computer generated, distorted" \
    --width 672 \
    --height 384 \
    --fps 15 \
    --num_frames 77 \
    --guidance_scale 7.5 \
    --seed 42 \
    --num_inference_steps 50 \
    --timestep_shift 10.0