export OPENAI_API_KEY=""
export OPENAI_API_BASE=''

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=claude-3-7-sonnet-20250219,max_frames_num=8 \
    --tasks robovqa \
    --batch_size 1 \
    --output_path /home/lx/embodied-eval/logs/logs_claude-3-7/