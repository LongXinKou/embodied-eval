
accelerate launch --num_processes=4 --main_process_port=12345 -m embodied_eval \
    --model llava_onevision \
    --model_args model_name_or_path=/data/klx/hf_model/llava-onevision-qwen2-7b-ov/,model_name=llava_qwen \
    --tasks where2place \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_llava_onevision