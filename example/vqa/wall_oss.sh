export OPENAI_API_KEY=""
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

 CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model wall_oss \
    --model_args model_name_or_path=/mnt/18T/klx/hf_model/wall-oss-fast/,max_num_frames=8 \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path /home/lx/embodied-eval/logs/logs_wall_oss/