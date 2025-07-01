export OPENAI_API_KEY=""
export OPENAI_API_BASE=''

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gemini-2.5-pro,max_frames_num=8 \
    --tasks vsibench \
    --batch_size 1 \
    --output_path logs/logs_gemini-2.5-pro/