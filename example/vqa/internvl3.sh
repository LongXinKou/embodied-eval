export OPENAI_API_KEY=''
export OPENAI_API_BASE=''
export CUDA_VISIBLE_DEVICES=0,1

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model internvl3 \
    --model_args model_name_or_path=/8T/klx/klx/hf_model/internvl3-8b/,num_frame=8,max_num=1 \
    --tasks erqa \
    --batch_size 1 \
    --output_path /home/lx/embodied-eval/logs/logs_internvl3