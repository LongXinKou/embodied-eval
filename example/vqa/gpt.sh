export OPENAI_API_KEY=""
export OPENAI_API_BASE=''

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-4o,max_frames_num=16 \
    --tasks robovqa \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_gpt-4o/