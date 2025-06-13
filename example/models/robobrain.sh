
accelerate launch --num_processes=2 --main_process_port=12345 -m embodied_eval \
    --model robobrain \
    --model_args model_name_or_path=/data/klx/hf_model/robobrain/,lora_id=/data/klx/hf_model/robobrain/affordance/ \
    --tasks where2place \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_robobrain/