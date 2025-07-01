export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=doubao-1.5-vision-pro-250328, \
    --tasks where2place \
    --batch_size 1 \
    --output_path logs/logs_seed1_5_vl_pro/