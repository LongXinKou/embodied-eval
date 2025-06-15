
accelerate launch --num_processes=4 --main_process_port=12347 -m embodied_eval \
    --model cosmos_reason1 \
    --model_args model_name_or_path=/data/klx/hf_model/cosmos-reason1-7B/,max_num_frames=8 \
    --tasks erqa \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_cosmos_reason1/