for p_date in `echo 2024-02-04`
do
	echo $p_date
	nohup sh export_data_from_hive_m2v.sh \
	"SELECT video_ceph_path, ori_caption_en, gen_mplug_caption_en, mplug_cogvlm_internlm_en, flow_score_top, width, height
	FROM mmu_vcg.m2v_meta_data_info_dt
	WHERE p_date = '$p_date'
	and duration >=2000 and width >=720 and height >=720 and watermark_score <0.35 and ocr_area <0.15 and grid_score <0.125 and porn_score <0.8 and nsfw_score <0.5 and data_source in (1 , 2 , 3 , 4 , 5 , 6);" ./threshold/${p_date}_all3_new.txt > logs/$p_date.log 2>&1 &

done