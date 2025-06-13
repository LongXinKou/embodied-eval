
# accelerate launch --num_processes=1 --main_process_port=12345 -m embodied_eval \
#     --model cosmos_reason1 \
#     --model_args model_name_or_path=/8T/klx/klx/hf_model/cosmos-reason1-7B/ \
#     --tasks where2place \
#     --batch_size 1 \
#     --output_path embodied-eval/logs/logs_cosmos_reason1_7B/

accelerate launch --num_processes=1 --main_process_port=12345 -m embodied_eval \
    --model cosmos_reason1 \
    --model_args model_name_or_path=/8T/klx/klx/hf_model/cosmos-reason1-7B/ \
    --tasks erqa \
    --batch_size 1 \
    --output_path embodied-eval/logs/logs_cosmos_reason1_7B