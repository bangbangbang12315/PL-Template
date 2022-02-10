python main.py  --gpus 0 \
                --batch_size 8 \
                --accumulate_grad_batches 16 \
                --precision 16 \
                --model_name GPT2 \
                --config_path ./pretrained/small_117M

