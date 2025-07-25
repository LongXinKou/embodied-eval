export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=hf_model/Qwen2_5-VL-32B-Instruct/,max_num_frames=8 \
    --tasks robovqa \
    --batch_size 1 \
    --output_path logs/logs_qwen2_5_vl/