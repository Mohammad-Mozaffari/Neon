GPUs=""
FUSION_FACTOR=1
GRID_SIZE=256
for GPU in 0 1 2 3
do
    GPUs="${GPUs} ${GPU}"
    GPUsCSV="${GPUs}, ${GPU}"
    nsys profile --force-overwrite true -o ./system/fusion_factor_${FUSION_FACTOR}_domain_size_${GRID_SIZE}_GPUs_$GPU ~/Neon/build/bin/fusion_map --fusion_factor $FUSION_FACTOR --domain_size $GRID_SIZE --num_blocks $FUSION_FACTOR --times 1 --warmup 0 --eval_scalability 0 --gpus $GPUs --run_baseline 0
    # ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2722 --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 --run_baseline 0
    for PROFILE_GPU in $(seq 0 "${GPU}")
    do
        ncu --kernel-regex execLambdaWithIterator_cuda --devices $PROFILE_GPU  -o ./compute/fusion_factor_${FUSION_FACTOR}_domain_size_${GRID_SIZE}_numGPUs_${GPU}_GPU_${PROFILE_GPU} --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 1 --times 1 --warmup 0 --eval_scalability 0 --gpus $GPUs --run_baseline 0
    done
done
# ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2722 --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 --run_baseline 0
# ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2170 --launch-count 1 -o ./compute/GPU3_1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 3 --run_baseline 0
# ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2287 --launch-count 1 -o ./compute/GPU3_0 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 3 --run_baseline 0
# ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2287 --launch-count 1 -o ./compute/GPU3_2 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 3 --run_baseline 0
# ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2289 --launch-count 1 -o ./compute/GPU3_3 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 3 --run_baseline 0


