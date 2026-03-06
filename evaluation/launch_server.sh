export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

python -m sglang.launch_server \
    --model-path /models/Qwen3.5-397B-A17B \
    --port 9000 \
    --tp-size 8 \
    --mem-fraction-static 0.9 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --disable-radix-cache \
    --cuda-graph-max-bs 64