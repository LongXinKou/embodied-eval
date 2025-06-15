
accelerate launch --num_processes=1 --main_process_port=12347 -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/data/klx/hf_model/Qwen2_5-VL-7B-Instruct/,max_num_frames=8 \
    --tasks vsibench \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_qwen2_5_vl/