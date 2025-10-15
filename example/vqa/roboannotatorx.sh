export OPENAI_API_KEY=""
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model roboannotatorx \
    --model_args model_name_or_path=/8T/klx/kisa-v2/work_dirs/roboannotatorx/roboannotatorx-7b-grid8-interval60-stage3-video-epoch-1/full_model/ \
    --evaluator eqa \
    --tasks vsibench \
    --batch_size 1 \
    --output_path /home/lx/embodied-eval/logs/logs_roboannotatorx/