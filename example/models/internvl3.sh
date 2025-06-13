
accelerate launch --num_processes=4 --main_process_port=12345 -m embodied_eval \
    --model internvl3 \
    --model_args model_name_or_path=/data/klx/hf_model/internvl3-8b/ \
    --tasks where2place \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_internvl3