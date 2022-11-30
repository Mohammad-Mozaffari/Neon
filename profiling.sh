DOMAIN_SIZE=512
for BLOCK_SIZE in "32,1,16" "32,16,1" "32,2,8"  #"32,1,32" #"128,1,4" "8,8,8" #"128,1,1"  "128,2,1" "128,4,1" "128,1,2" "128,2,2" #"256,1,1" "64,4,1" "32,8,1" "16,16,1"
do
    BLOCK_SIZE_STR=
    sed -i "s/block_size(.*,.*,.*);/block_size(${BLOCK_SIZE});/g" ../libNeonSolver/src/linear/matvecs/LaplacianMatVec.cu
    for FILE_NAME in fused baseline
    do
        if [ $FILE_NAME = fused ]
        then
            sed -i "s/#define fusion .*/#define fusion MAP_STENCIL/g" ../libNeonSolver/src/linear/krylov/CG.cpp
            make
            # ./bin/solverPt_Poisson --domain_size $DOMAIN_SIZE
            nsys profile --force-overwrite true -o "./prof/fused_${BLOCK_SIZE}" ./bin/solverPt_Poisson --domain_size $DOMAIN_SIZE
            ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 6 --launch-count 1 --set full -f -o "./prof/fused_${BLOCK_SIZE}" "./bin/solverPt_Poisson"  --domain_size $DOMAIN_SIZE
        else
            sed -i "s/#define fusion .*/#define fusion BASELINE/g" ../libNeonSolver/src/linear/krylov/CG.cpp
            make
            # ./bin/solverPt_Poisson --domain_size $DOMAIN_SIZE
            nsys profile --force-overwrite true -o "./prof/baseline_${BLOCK_SIZE}" ./bin/solverPt_Poisson --domain_size $DOMAIN_SIZE
            ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 7 --launch-count 1 --set full -f -o "./prof/baseline_map_${BLOCK_SIZE}" "./bin/solverPt_Poisson" --domain_size $DOMAIN_SIZE
            ncu --kernel-regex execLambdaWithIterator_cuda --launch-skip 8 --launch-count 1 --set full -f -o "./prof/baseline_stencil_${BLOCK_SIZE}" "./bin/solverPt_Poisson" --domain_size $DOMAIN_SIZE
        fi
    done
done
