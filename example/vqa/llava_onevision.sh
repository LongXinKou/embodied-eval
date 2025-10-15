export OPENAI_API_KEY=''
export OPENAI_API_BASE=''
export CUDA_VISIBLE_DEVICES=1,2,3
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model llava_onevision \
    --model_args model_name_or_path=/data/klx/hf_model/llava-onevision-qwen2-7b-ov/,model_name=llava_qwen,max_num_frames=16 \
    --tasks openeqa-emeqa \
    --batch_size 1 \
    --output_path /data/klx/embodied-eval/logs/logs_llava_onevision