export OPENAI_API_KEY="sk-BR51CvrlUzNd8QSVmpwG8ZiIFT5l2HX11o8ddHdYLmB0chku"
export OPENAI_API_BASE="https://yunwu.ai/v1"

python -m embodied_eval \
    --model openai_compatible \
    --model_args model_name_or_path=gemini-2.0-flash, \
    --tasks where2place \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_robobrain/