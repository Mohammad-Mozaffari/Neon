# GPUs=""
# FUSION_FACTOR=1
# GRID_SIZE=256
# for GPU in 0 1 2 3
# do
#     GPUs="${GPUs} ${GPU}"
#     nsys profile --force-overwrite true -o ./single_GPU_samples/fusion_factor_${FUSION_FACTOR}_domain_size_${GRID_SIZE}_GPUs_$GPU ~/Neon/build/bin/fusion_map --fusion_factor $FUSION_FACTOR --domain_size $GRID_SIZE --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus $GPU --run_baseline 0
#     ncu --kernel-name execLambdaWithIterator_cuda --launch-skip 2722 --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 --run_baseline 0
# done
# ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 2722 --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 --run_baseline 0
sudo ncu --kernel-regex execLambdaWithIterator_cuda -o /home/mohammad/Neon/optimization/fusion/scripts/compute/GPU0_prof --launch-skip 548 --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 --run_baseline 0
sudo ncu --kernel-regex execLambdaWithIterator_cuda -o /home/mohammad/Neon/optimization/fusion/scripts/compute/GPU2-prof --launch-skip 575 --launch-count 1 "/home/mohammad/Neon/build/bin/fusion_map" --fusion_factor 1 --domain_size 256 --num_blocks 512 --times 10 --warmup 2 --eval_scalability 0 --gpus 0 1 2 --run_baseline 0