#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#include "Neon/Neon.h"

#include "Neon/Report.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/set/DevSet.h"
#include "Neon/skeleton/Skeleton.h"
#include "Poisson.h"
#include "gtest/gtest.h"
#include "Neon/core/types/DataUse.h"


using namespace Neon::set;
using namespace Neon::solver;
using namespace Neon::domain;


using Grid = Neon::domain::dGrid;
using Real = double;
using Field = typename Grid::template Field<Real>;
using bdField = typename Grid::template Field<int8_t>;


#define DOMAIN_SIZE 128
#define BLOCK_X 16
#define BLOCK_Y 01
#define BLOCK_Z 16


template <typename Grid, typename Real>
inline Neon::set::Container load_shared_memory(const Field&   input,
                                                                      const bdField& boundary)
{
    // Real stepSize = m_h;
    Neon::index_3d block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);

    auto cont = input.getGrid().getContainer("Laplacian",  block_size, 0, [&](Neon::set::Loader& L) {
        auto& inp = L.load(input, Neon::Compute::STENCIL);
        auto& bnd = L.load(boundary);

        // Precompute 1/h^2
        // const Real invh2 = Real(1.0) / (stepSize * stepSize);

        return [=] NEON_CUDA_DEVICE_ONLY(const typename Grid::template Field<Real>::Cell& cell) mutable {
            // const int cardinality = inp.cardinality();

            // if(!(blockIdx.x ==0 && blockIdx.y == 327 && blockIdx.z == 10))
            // {
            //    return;
            // }
            // Iterate through each element's cardinality
            constexpr int c = 0;
            const Real center = inp(cell, c);
            // if (bnd(cell, c) == 0) {
            //     ;
            // } else {
                Real error(0.0);
                const Real defaultVal{0};

                // Laplacian stencil operates on 6 neighbors (assuming 3D)
                constexpr int x_size = BLOCK_X + 2, y_size = BLOCK_Y + 2, z_size = BLOCK_Z + 2;
                int x = threadIdx.x + 1, y = threadIdx.y + 1, z = threadIdx.z + 1;

                int threadsPerBlock  = blockDim.x * blockDim.y + blockDim.z;

                int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y + threadIdx.z * blockDim.x * blockDim.y;

                int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y  + gridDim.x * gridDim.y * blockIdx.z;

                int global_idx = blockNumInGrid * threadsPerBlock + threadNumInBlock;
                __shared__ Real vals[x_size][y_size][z_size];
                // __shared__ bool valid[x_size][y_size][z_size];
                
                // int idx = x + y * x_size + z * x_size * y_size;
                // vals[x][y][z] = center;

                if(x == 1) {
                    auto ngh = inp.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                    vals[x - 1][y][z] = ngh.value;
                    // valid[idx - 1] = ngh.isValid;
                }
                if(x == BLOCK_X) {
                    auto ngh = inp.template nghVal<1, 0, 0>(cell, c, defaultVal);
                    vals[x + 1][y][z] = ngh.value;
                    // valid[idx + 1] = ngh.isValid;
                }
                if(y == 1) {
                    auto ngh = inp.template nghVal<0, -1, 0>(cell, c, defaultVal);
                    vals[x][y - 1][z] = ngh.value;
                    // valid[idx - x_size] = ngh.isValid;
                }
                if(y == BLOCK_Y) {
                    auto ngh = inp.template nghVal<0, 1, 0>(cell, c, defaultVal);
                    vals[x][y + 1][z] = ngh.value;
                    // valid[idx + x_size] = inp.template nghVal<0, 1, 0>(cell, c, defaultVal).isValid;
                }
                if(z == 1 || z == 2) {
                    auto ngh = inp.template nghVal<0, 0, -1>(cell, c, defaultVal);
                    vals[x][y][z - 1] = ngh.value;
                    // valid[idx - x_size * y_size] = ngh.isValid;
                }
                // if(z == BLOCK_Z) {
                    auto ngh = inp.template nghVal<0, 0, 1>(cell, c, defaultVal);
                    vals[x][y][z + 1] = ngh.value;
                    // valid[idx + x_size * y_size] = ngh.isValid;
                // }


                // auto ngh = inp.template nghVal<0, 0, -1>(cell, c, defaultVal);
                // vals[x][y][z - 1] = ngh.value;

                // auto ngh = inp.template nghVal<0, 0, 1>(cell, c, defaultVal);
                // vals[x][y][z + 1] = ngh.value;

                
                // valid[idx s- x_size * y_size] = ngh.isValid;

                // vals[x][y][z] = (Real) (threadIdx.x + threadIdx.y + threadIdx.z);
                // valid[x][y][z] = true;

                // __syncthreads();

                // if(x == 1) {
                //     auto ngh = inp.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                //     vals[idx - 1] = ngh.value;
                //     valid[idx - 1] = ngh.isValid;
                // }
                // if(x == BLOCK_X) {
                //     auto ngh = inp.template nghVal<1, 0, 0>(cell, c, defaultVal);
                //     vals[idx + 1] = ngh.value;
                //     valid[idx + 1] = ngh.isValid;
                // }
                // // if(y == 1) {
                // //     auto ngh = inp.template nghVal<0, -1, 0>(cell, c, defaultVal);
                // //     vals[idx - x_size] = ngh.value;
                // //     valid[idx - x_size] = ngh.isValid;
                // // }
                // // if(y == BLOCK_Y) {
                // //     auto ngh = inp.template nghVal<0, 1, 0>(cell, c, defaultVal);
                // //     vals[idx + x_size] = ngh.value;
                // //     valid[idx + x_size] = inp.template nghVal<0, 1, 0>(cell, c, defaultVal).isValid;
                // // }
                // // if(z == 1) {
                //     auto ngh = inp.template nghVal<0, 0, -1>(cell, c, defaultVal);
                //     vals[idx - x_size * y_size] = ngh.value;
                //     valid[idx - x_size * y_size] = ngh.isValid;
                // // }
                // // if(z == BLOCK_Z) {
                //     ngh = inp.template nghVal<0, 0, 1>(cell, c, defaultVal);
                //     vals[idx + x_size * y_size] = ngh.value;
                //     valid[idx + x_size * y_size] = ngh.isValid;
                // // }
                __syncthreads();
                // if(x==1 || x==BLOCK_X  || z==1 || z==BLOCK_Z) {
                //     return;
                // }
                error = (vals[x - 1][y][z] - inp.template nghVal<-1, 0, 0>(cell, c, defaultVal).value);
                if (error * error > 1e-5)
                {
                    printf("Error -x: %d, %d, %d, %lf, %lf, %lf, Global Index: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[x - 1][y][z], inp.template nghVal<-1, 0, 0>(cell, c, defaultVal).value, center, global_idx);
                }
                error = (vals[x + 1][y][z] - inp.template nghVal<1, 0, 0>(cell, c, defaultVal).value);
                if (error * error > 1e-5)
                {
                    printf("Error +x: %d, %d, %d, %lf, %lf, %lf, Global Index: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[x + 1][y][z], inp.template nghVal<1, 0, 0>(cell, c, defaultVal).value, center, global_idx);
                }
                // error = (vals[idx - x_size] - inp.template nghVal<0, -1, 0>(cell, c, defaultVal).value);
                // if (error * error > 1e-5)
                // {
                //     printf("Error -y: %d, %d, %d, %lf, %lf\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[idx + x_size * y_size], inp.template nghVal<0, -1, 0>(cell, c, defaultVal).value);
                // }
                // error = (vals[idx + x_size] - inp.template nghVal<0, 1, 0>(cell, c, defaultVal).value);
                // if (error * error > 1e-5)
                // {
                //     printf("Error +y: %d, %d, %d, %lf, %lf\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[idx + x_size * y_size], inp.template nghVal<0, 1, 0>(cell, c, defaultVal).value);
                // }

                error = (vals[x][y][z - 1] - inp.template nghVal<0, 0, -1>(cell, c, defaultVal).value);
                if (error * error > 1e-5)
                {
                    printf("Error -z: Block Index: %d, %d, %d, Thread Index: %d, %d, %d, %lf, %lf, %lf, Global Index: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, vals[x][y][z - 1], inp.template nghVal<0, 0, -1>(cell, c, defaultVal).value, vals[x][y][z], global_idx);
                }
                error = (vals[x][y][z + 1] - inp.template nghVal<0, 0, 1>(cell, c, defaultVal).value);
                if (error * error > 1e-5)
                {
                    printf("Error +z: Block Index: %d, %d, %d, Thread Index: %d, %d, %d, %lf, %lf, %lf, Global Index: %d\n",  blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, vals[x][y][z + 1], inp.template nghVal<0, 0, 1>(cell, c, defaultVal).value, vals[x][y][z], global_idx);
                }
                

                // printf("IDX: %d, %d, %d Done\n", threadIdx.x, threadIdx.y, threadIdx.z);

            // }
        };
    });
    return cont;
}


TEST(shared_mem_test, neighbour_loading)
{

    Neon::init();
    Neon::Backend backend = [] {
        Neon::init();
        // auto runtime = Neon::Runtime::openmp;
        auto runtime = Neon::Runtime::stream;
        // We are overbooking XPU 0 three times
        std::vector<int> xpuIds{0, 0, 0};
        Neon::Backend    backend(xpuIds, runtime);
        // Printing some information
        NEON_INFO(backend.toString());
        return backend;
    }();

    Grid grid = createGrid<Grid>(backend, DOMAIN_SIZE);

    auto u = grid.template newField<Real>("u", 1, Real(0), Neon::DataUse::IO_COMPUTE);
    auto bd = grid.template newField<int8_t>("bd", 1, int8_t(0), Neon::DataUse::IO_COMPUTE);

    const Neon::index_3d dims = grid.getDimension();

    u.forEachActiveCell([&](const Neon::index_3d& idx,
                                           const int& card, Real& val) {
        // val = idx.x + idx.y * 1000 + idx.z * 1000000;
        val = (idx.x % BLOCK_X) + (idx.y % BLOCK_Y) + (idx.z % BLOCK_Z);
    });

    bd.forEachActiveCell([dims](const Neon::index_3d& idx, const int& /*card*/, int8_t& val) {
        val = (idx.z == 0 || idx.z == dims.z - 1) ? Neon::solver::BoundaryCondition::Fixed : Neon::solver::BoundaryCondition::Free;
    });
    

    u.updateCompute(0);
    bd.updateCompute(0);

    // auto L = std::make_shared<Neon::solver::FusedLaplacianMatVec<Grid, Real>>(Real(1.0));

    Neon::skeleton::Skeleton cgIter(backend);

    cgIter.sequence({load_shared_memory<Grid, Real>(u, bd)}, "matVec");

    cgIter.run();
    printf("Done\n");



}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}