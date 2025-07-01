export OPENAI_API_KEY=""
export OPENAI_API_BASE=''

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=o4-mini,max_frames_num=8 \
    --tasks robovqa \
    --batch_size 1 \
    --output_path logs/logs_o4-mini/