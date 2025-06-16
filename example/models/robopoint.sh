
accelerate launch --num_processes=4 --main_process_port=12346 -m embodied_eval \
    --model robopoint \
    --model_args model_name_or_path=/data/klx/hf_model/robopint/ \
    --tasks where2place \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_robopoint