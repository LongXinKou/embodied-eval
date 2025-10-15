export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model robobrain2 \
    --model_args model_name_or_path=/8T/klx/klx/hf_model/robobrain2-32b/,max_num_frames=16,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path /home/lx/embodied-eval/logs/logs_robobrain2/