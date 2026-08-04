input_tokens=8192 #8192, 16384, 32768, 65536, 131072, 262144, 524288
output_tokens=1
export SGLANG_TORCH_PROFILER_DIR="./mimo_pro_2.5_aiter_prefill_profile_8k_opt"

python3 -m sglang.bench_serving \
    --backend sglang \
    --model /cephfs/MiMo-V2.5-Pro/ \
    --host 0.0.0.0 \
    --port 30001 \
    --dataset-name random \
    --random-input-len ${input_tokens} \
    --random-output-len ${output_tokens} \
    --random-range-ratio 1.0 \
    --flush-cache \
    --seed 12345 \
    --num-prompts 16 \
    --warmup-requests 4 \
    --max-concurrency 4 \
    --profile

