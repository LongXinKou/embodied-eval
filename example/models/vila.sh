export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model vila \
    --model_args model_name_or_path=hf_model/vila1_5-13B/,max_num_frames=8,model_name=VILA1.5-13B \
    --tasks erqa \
    --batch_size 1 \
    --output_path logs/logs_vila