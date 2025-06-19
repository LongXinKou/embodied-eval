
accelerate launch --num_processes=4 --main_process_port=12236 -m embodied_eval \
    --model physvlm \
    --model_args model_name_or_path=/data/klx/hf_model/PhysVLM-3B/,max_num_frames=8 \
    --tasks robovqa \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_physvlm