#include "Neon/core/core.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/solver/linear/matvecs/LaplacianMatVec.h"

namespace Neon {
namespace solver {

#define SHARED_MEM 0
#define BLOCK_X 32
#define BLOCK_Y 01
#define BLOCK_Z 01

template <typename Grid, typename Real>
inline Neon::set::Container LaplacianMatVec<Grid, Real>::matVec(const Field&   input,
                                                                      const bdField& boundary,
                                                                      Field&         output)
{
    Real stepSize = m_h;
    Neon::index_3d block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);

    auto cont = input.getGrid().getContainer("Laplacian",  block_size, 0, [&, stepSize](Neon::set::Loader& L) {
        auto& inp = L.load(input, Neon::Compute::STENCIL);
        auto& bnd = L.load(boundary);
        auto& out = L.load(output);

        // Precompute 1/h^2
        const Real invh2 = Real(1.0) / (stepSize * stepSize);

        return [=] NEON_CUDA_DEVICE_ONLY(const typename Grid::template Field<Real>::Cell& cell) mutable {
            const int cardinality = inp.cardinality();

            
            // Iterate through each element's cardinality
            constexpr int c = 0;
            // for (int c = 0; c < cardinality; ++c) {
                const Real center = inp(cell, c);
                if (bnd(cell, c) == 0) {
                    out(cell, c) = center;
                } else {
                    Real error(0.0);
                    Real       sum(0.0);
                    int           numNeighb = 0;
                    const Real defaultVal{0};

                    auto checkNeighbor = [&sum, &numNeighb](Neon::domain::NghInfo<Real>& neighbor) {
                        if (neighbor.isValid) {
                            ++numNeighb;
                            sum += neighbor.value;
                        }
                    };
                    // Laplacian stencil operates on 6 neighbors (assuming 3D)
                    if constexpr (std::is_same<Grid, Neon::domain::internal::eGrid::eGrid>::value) {
                        for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                            auto neighbor = inp.nghVal(cell, nghIdx, c, defaultVal);
                            checkNeighbor(neighbor);
                        }
                    } else if(SHARED_MEM == 0) {
                        // typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        // ngh.x = 1;
                        // ngh.y = 0;
                        // ngh.z = 0;
                        auto neighbor = inp.template nghVal<1, 0, 0>(cell, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-x
                        // ngh.x = -1;
                        // ngh.y = 0;
                        // ngh.z = 0;
                        neighbor = inp.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                        checkNeighbor(neighbor);

                        //+y
                        // ngh.x = 0;
                        // ngh.y = 1;
                        // ngh.z = 0;
                        neighbor = inp.template nghVal<0, 1, 0>(cell, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-y
                        // ngh.x = 0;
                        // ngh.y = -1;
                        // ngh.z = 0;
                        neighbor = inp.template nghVal<0, -1, 0>(cell, c, defaultVal);
                        checkNeighbor(neighbor);

                        //+z
                        // ngh.x = 0;
                        // ngh.y = 0;
                        // ngh.z = 1;
                        neighbor = inp.template nghVal<0, 0, 1>(cell, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-z
                        // ngh.x = 0;
                        // ngh.y = 0;
                        // ngh.z = -1;
                        neighbor = inp.template nghVal<0, 0, -1>(cell, c, defaultVal);
                        checkNeighbor(neighbor);
                    } else {
                        constexpr int x_size = BLOCK_X + 2, y_size = BLOCK_Y + 2, z_size = BLOCK_Z + 2;
                        int x = threadIdx.x + 1, y = threadIdx.y + 1, z = threadIdx.z + 1;

                        __shared__ Real vals[x_size * y_size * z_size];
                        __shared__ bool valid[x_size * y_size * z_size];
                        
                        int idx = x + y * x_size + z * x_size * y_size;
                        vals[idx] = center;
                        valid[idx] = true;

                        if(x == 1) {
                            auto ngh = inp.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                            vals[idx - 1] = ngh.value;
                            valid[idx - 1] = ngh.isValid;
                        }
                        if(x == BLOCK_X) {
                            auto ngh = inp.template nghVal<1, 0, 0>(cell, c, defaultVal);
                            vals[idx + 1] = ngh.value;
                            valid[idx + 1] = ngh.isValid;
                        }
                        if(y == 1) {
                            auto ngh = inp.template nghVal<0, -1, 0>(cell, c, defaultVal);
                            vals[idx - x_size] = ngh.value;
                            valid[idx - x_size] = ngh.isValid;
                        }
                        if(y == BLOCK_Y) {
                            auto ngh = inp.template nghVal<0, 1, 0>(cell, c, defaultVal);
                            vals[idx + x_size] = ngh.value;
                            valid[idx + x_size] = inp.template nghVal<0, 1, 0>(cell, c, defaultVal).isValid;
                        }
                        // if(z == 1) {
                            // typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, -1);
                            // vals[idx - x_size * y_size] = inp.nghVal(cell, ngh, c, defaultVal).value;
                            // valid[idx - x_size * y_size] = inp.nghVal(cell, ngh, c, defaultVal).isValid;
                            auto ngh = inp.template nghVal<0, 0, -1>(cell, c, defaultVal);
                            vals[idx - x_size * y_size] = ngh.value;
                            valid[idx - x_size * y_size] = ngh.isValid;
                        // }
                        // if(z == BLOCK_Z) {
                            // typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, +1);
                            // vals[idx - x_size * y_size] = inp.nghVal(cell, ngh, c, defaultVal).value;
                            // valid[idx - x_size * y_size] = inp.nghVal(cell, ngh, c, defaultVal).isValid;
                            ngh = inp.template nghVal<0, 0, 1>(cell, c, defaultVal);
                            vals[idx + x_size * y_size] = ngh.value;
                            valid[idx + x_size * y_size] = ngh.isValid;
                        // }

                        
                        // int idx = (x - 1) + (y - 1) * x_size + (z - 1) * x_size * y_size;
                        // auto neighbor = inp.template nghVal<-1, -1, -1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();

                        // idx = (x - 1) + (y - 1) * x_size + (z + 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<-1, -1, +1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();
                        
                        // idx = (x - 1) + (y + 1) * x_size + (z - 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<-1, +1, -1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();

                        // idx = (x - 1) + (y + 1) * x_size + (z + 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<-1, +1, +1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();
                        
                        // idx = (x + 1) + (y - 1) * x_size + (z - 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<+1, -1, -1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();

                        // idx = (x + 1) + (y - 1) * x_size + (z + 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<+1, -1, +1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();

                        // idx = (x + 1) + (y + 1) * x_size + (z - 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<+1, +1, -1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;
                        // __syncthreads();

                        // idx = (x + 1) + (y + 1) * x_size + (z + 1) * x_size * y_size;
                        // neighbor = inp.template nghVal<+1, +1, +1>(cell, c, defaultVal);
                        // vals[idx] = neighbor.value;
                        // valid[idx] = neighbor.isValid;

                        __syncthreads();

                        // if(threadIdx.z == 14)
                        // {
                        //     printf("Check: %d, %d, %d, %f, %f\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[idx + x_size * y_size], inp.template nghVal<0, 0, +1>(cell, c, defaultVal).value);
                        // }
                        // if(threadIdx.z == 15)
                        // {
                        //     printf("Check: %d, %d, %d, %f, %f\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[idx], inp.template nghVal<0, 0, 0>(cell, c, defaultVal).value);
                        // }

                        
                        // typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, +1);
                        // // error = (vals[idx + x_size * y_size] - inp.nghVal(cell, ngh, c, defaultVal).value) * (vals[idx + x_size * y_size] - inp.nghVal(cell, ngh, c, defaultVal).value);
                        
                        // error = (vals[idx + x_size * y_size] - inp.template nghVal<0, 0, +1>(cell, c, defaultVal).value) * (vals[idx + x_size * y_size] - inp.template nghVal<0, 0, +1>(cell, c, defaultVal).value);
                        // // error = (vals[idx - x_size] - inp.template nghVal<0, -1, 0>(cell, c, defaultVal).value) * (vals[idx - x_size] - inp.template nghVal<0, -1, 0>(cell, c, defaultVal).value);
                        // if(error < 1e-5)
                        // {

                        
                            // idx = x + 1 + y * x_size + z * x_size * y_size;
                            sum += vals[idx + 1];
                            if (valid[idx + 1]) {
                                ++numNeighb;
                            }

                            // idx = x - 1 + y * x_size + z * x_size * y_size;
                            sum += vals[idx - 1];
                            if (valid[idx - 1]) {
                                ++numNeighb;
                            }

                            // idx = x + (y + 1) * x_size + z * x_size * y_size;
                            sum += vals[idx + x_size];
                            if (valid[idx + x_size]) {
                                ++numNeighb;
                            }

                            // idx = x + (y - 1) * x_size + z * x_size * y_size;
                            sum += vals[idx - x_size];
                            if (valid[idx - x_size]) {
                                ++numNeighb;
                            }


                            sum += vals[idx + x_size * y_size];
                            if (valid[idx + x_size * y_size]) {
                                ++numNeighb;
                            }


                            sum += vals[idx - x_size * y_size];
                            if (valid[idx - x_size * y_size]) {
                                ++numNeighb;
                            }

                            // ngh.z = -1;
                            // sum += inp.nghVal(cell, ngh, c, defaultVal).value;
                            // if (inp.nghVal(cell, ngh, c, defaultVal).isValid) {
                            //     ++numNeighb;
                            // }

                            // ngh.z = +1;
                            // sum += inp.nghVal(cell, ngh, c, defaultVal).value;
                            // if (inp.nghVal(cell, ngh, c, defaultVal).isValid) {
                            //     ++numNeighb;
                            // }
                            
                            // idx = x + y * x_size + (z + 1) * x_size * y_size;
                            // sum += inp.template nghVal<0, 0, +1>(cell, c, defaultVal).value;
                            // if (inp.template nghVal<0, 0, +1>(cell, c, defaultVal).isValid) {
                            //     ++numNeighb;
                            // }

                            // idx = x + y * x_size + (z - 1) * x_size * y_size;
                            // sum += inp.template nghVal<0, 0, -1>(cell, c, defaultVal).value;
                            // if (inp.template nghVal<0, 0, -1>(cell, c, defaultVal).isValid) {
                            //     ++numNeighb;
                            // }
                        // }
                        // else
                        // {
                        //     // if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                        //     // {
                        //         // printf("Error: %d, %d, %d, %f, %f\n", threadIdx.x, threadIdx.y, threadIdx.z, vals[idx + x_size * y_size], inp.template nghVal<0, 0, +1>(cell, c, defaultVal).value);
                        //     // }
                        //     sum = -1000.0;
                        // }
                    }
                    out(cell, c) = (-sum + static_cast<Real>(numNeighb) * center) * invh2;
                }
            // }
        };
    });
    return cont;
}

// Template instantiations
template class LaplacianMatVec<Neon::domain::eGrid, double>;
template class LaplacianMatVec<Neon::domain::eGrid, float>;
template class LaplacianMatVec<Neon::domain::dGrid, double>;
template class LaplacianMatVec<Neon::domain::dGrid, float>;
// template class LaplacianMatVec<Neon::domain::bGrid, double>;
// template class LaplacianMatVec<Neon::domain::bGrid, float>;

// p2 := r + beta * p1
// s := A (r + beta * p1)
// --------------------------------------
// output1 := input1 + beta * input2
// output2 := A (input1 + beta * input2)
template <typename Grid, typename Real>
inline Neon::set::Container FusedLaplacianMatVec<Grid, Real>::fusedMatVec(const Field&   input1,
                                                                      const Field& input2,
                                                                      const Real& delta_new,
                                                                      const Real& delta_old,
                                                                      const bdField& boundary,
                                                                      Field&         output1,
                                                                      Field&         output2,
                                                                      Neon::template PatternScalar<Real>& scalar)
{
    Real stepSize = this->stepSize();
    Neon::index_3d block_size(16,1,16);

    auto cont = input1.getGrid().getContainer("Fused_Map_and_Laplacian", block_size, (BLOCK_X + 2) * (BLOCK_Y + 2) * (BLOCK_Z + 2) * (sizeof(bool) + 2 * sizeof(Real)), [&, stepSize](Neon::set::Loader& L) {
        auto& inp1 = L.load(input1, Neon::Compute::STENCIL);
        auto& inp2 = L.load(input2, Neon::Compute::STENCIL);
        auto& bnd = L.load(boundary);
        auto& out1 = L.load(output1);
        auto& out2 = L.load(output2);
        auto& scal = L.load(scalar);


        Real beta;
        if (std::abs(delta_old) > std::numeric_limits<Real>::epsilon()) {
            beta = delta_new / delta_old;
        } else {
            beta = 0;
        }

        // Precompute 1/h^2
        const Real invh2 = Real(1.0) / (stepSize * stepSize);

        return [=] NEON_CUDA_DEVICE_ONLY(const typename Grid::template Field<Real>::Cell& cell) mutable { //(int streamIdx, Neon::DataView dataView, const typename Grid::template Field<Real>::Cell& cell) mutable {
            const int cardinality = inp1.cardinality();


            
            // Iterate through each element's cardinality
            constexpr int c = 0;
            // for (int c = 0; c < cardinality; ++c) {
                const Real center1 = inp1(cell, c);
                const Real center2 = inp2(cell, c);

                const Real center = center1 + beta * center2;
                out1(cell, c) = center;
                if (bnd(cell, c) == 0) {
                    out1(cell, c) = center1; //DOUBLE CHECK
                    out2(cell, c) = center2;
                } 
                else {
                    Real       sum(0.0);
                    int           numNeighb = 0;
                    const Real defaultVal{0};

                    // auto checkNeighbor = [&sum, &numNeighb, &beta](Neon::domain::NghInfo<Real>& neighbor1, Neon::domain::NghInfo<Real>& neighbor2) {
                    //     if (neighbor1.isValid) {
                    //         ++numNeighb;
                    //         sum += neighbor1.value + beta * neighbor2.value;
                    //     }
                    // };
                    // Laplacian stencil operates on 6 neighbors (assuming 3D)
                    if constexpr (std::is_same<Grid, Neon::domain::internal::eGrid::eGrid>::value) {
                        for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                            auto neighbor1 = inp1.nghVal(cell, nghIdx, c, defaultVal);
                            auto neighbor2 = inp2.nghVal(cell, nghIdx, c, defaultVal);
                            sum += neighbor1.value + beta * neighbor2.value;
                            numNeighb += (int) neighbor1.isValid;
                        }
                    } else if(SHARED_MEM == 0) {
                        // typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        // ngh.x = 1;
                        // ngh.y = 0;
                        // ngh.z = 0;
                        Neon::domain::NghInfo<Real> neighbor1 = inp1.template nghVal<1, 0, 0>(cell, c, defaultVal);
                        Neon::domain::NghInfo<Real> neighbor2 = inp2.template nghVal<1, 0, 0>(cell, c, defaultVal);
                        sum += neighbor1.value + beta * neighbor2.value;
                        numNeighb += (int) neighbor1.isValid;

                        //-x
                        // ngh.x = -1;
                        // ngh.y = 0;
                        // ngh.z = 0;
                        neighbor1 = inp1.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                        neighbor2 = inp2.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                        sum += neighbor1.value + beta * neighbor2.value;
                        numNeighb += (int) neighbor1.isValid;

                        //+y
                        // ngh.x = 0;
                        // ngh.y = 1;
                        // ngh.z = 0;
                        neighbor1 = inp1.template nghVal<0, 1, 0>(cell, c, defaultVal);
                        neighbor2 = inp2.template nghVal<0, 1, 0>(cell, c, defaultVal);
                        sum += neighbor1.value + beta * neighbor2.value;
                        numNeighb += (int) neighbor1.isValid;

                        //-y
                        // ngh.x = 0;
                        // ngh.y = -1;
                        // ngh.z = 0;
                        neighbor1 = inp1.template nghVal<0, -1, 0>(cell, c, defaultVal);
                        neighbor2 = inp2.template nghVal<0, -1, 0>(cell, c, defaultVal);
                        sum += neighbor1.value + beta * neighbor2.value;
                        numNeighb += (int) neighbor1.isValid;

                        //+z
                        // ngh.x = 0;
                        // ngh.y = 0;
                        // ngh.z = 1;
                        neighbor1 = inp1.template nghVal<0, 0, 1>(cell, c, defaultVal);
                        neighbor2 = inp2.template nghVal<0, 0, 1>(cell, c, defaultVal);
                        sum += neighbor1.value + beta * neighbor2.value;
                        numNeighb += (int) neighbor1.isValid;

                        //-z
                        // ngh.x = 0;
                        // ngh.y = 0;
                        // ngh.z = -1;
                        neighbor1 = inp1.template nghVal<0, 0, -1>(cell, c, defaultVal);
                        neighbor2 = inp2.template nghVal<0, 0, -1>(cell, c, defaultVal);
                        sum += neighbor1.value + beta * neighbor2.value;
                        numNeighb += (int) neighbor1.isValid;
                    } 
                    // else
                    // {
                    //     constexpr int x_size = BLOCK_X + 2, y_size = BLOCK_Y + 2, z_size = BLOCK_Z + 2;
                    //     int x = threadIdx.x + 1, y = threadIdx.y + 1, z = threadIdx.z + 1;

                    //     __shared__ Real vals1[x_size * y_size * z_size];
                    //     __shared__ Real vals2[x_size * y_size * z_size];
                    //     __shared__ bool valid[x_size * y_size * z_size];
                        
                    //     int idx = x + y * x_size + z * x_size * y_size;
                    //     vals1[idx] = center1;
                    //     vals2[idx] = center2;
                    //     valid[idx] = true;

                    //     if(x == 1) {
                    //         auto ngh1 = inp1.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                    //         auto ngh2 = inp2.template nghVal<-1, 0, 0>(cell, c, defaultVal);
                    //         vals1[idx - 1] = ngh1.value;
                    //         vals2[idx - 1] = ngh2.value;
                    //         valid[idx - 1] = ngh1.isValid;
                    //     }
                    //     if(x == BLOCK_X)
                    //     {
                    //         auto ngh1 = inp1.template nghVal<1, 0, 0>(cell, c, defaultVal);
                    //         auto ngh2 = inp2.template nghVal<1, 0, 0>(cell, c, defaultVal);
                    //         vals1[idx + 1] = ngh1.value;
                    //         vals2[idx + 1] = ngh2.value;
                    //         valid[idx + 1] = ngh1.isValid;
                    //     }
                    //     if(y == 1) {
                    //         auto ngh1 = inp1.template nghVal<0, -1, 0>(cell, c, defaultVal);
                    //         auto ngh2 = inp2.template nghVal<0, -1, 0>(cell, c, defaultVal);
                    //         vals1[idx - x_size] = ngh1.value;
                    //         vals2[idx - x_size] = ngh2.value;
                    //         valid[idx - x_size] = ngh1.isValid;
                    //     }
                    //     if(y == BLOCK_Y)
                    //     {
                    //         auto ngh1 = inp1.template nghVal<0, 1, 0>(cell, c, defaultVal);
                    //         auto ngh2 = inp2.template nghVal<0, 1, 0>(cell, c, defaultVal);
                    //         vals1[idx + x_size] = ngh1.value;
                    //         vals2[idx + x_size] = ngh2.value;
                    //         valid[idx + x_size] = ngh1.isValid;
                    //     }
                    //     auto ngh1 = inp1.template nghVal<0, 0, -1>(cell, c, defaultVal);
                    //     auto ngh2 = inp2.template nghVal<0, 0, -1>(cell, c, defaultVal);
                    //     vals1[idx - x_size * y_size] = ngh1.value;
                    //     vals2[idx - x_size * y_size] = ngh2.value;
                    //     valid[idx - x_size * y_size] = ngh1.isValid;


                    //     ngh1 = inp1.template nghVal<0, 0, 1>(cell, c, defaultVal);
                    //     ngh2 = inp2.template nghVal<0, 0, 1>(cell, c, defaultVal);
                    //     vals1[idx + x_size * y_size] = ngh1.value;
                    //     vals2[idx + x_size * y_size] = ngh2.value;
                    //     valid[idx + x_size * y_size] = ngh1.isValid;

                    //     __syncthreads();

                    //     sum += vals1[idx + 1] + beta * vals2[idx + 1];
                    //     if (valid[idx + 1]) {
                    //         ++numNeighb;
                    //     }
                        
                    //     sum += vals1[idx - 1] + beta * vals2[idx - 1];
                    //     if (valid[idx - 1]) {
                    //         ++numNeighb;
                    //     }
                        
                    //     sum += vals1[idx + x_size] + beta * vals2[idx + x_size];
                    //     if (valid[idx + x_size]) {
                    //         ++numNeighb;
                    //     }
                        
                    //     sum += vals1[idx - x_size] + beta * vals2[idx - x_size];
                    //     if (valid[idx - x_size]) {
                    //         ++numNeighb;
                    //     }

                    //     sum += vals1[idx + x_size * y_size] + beta * vals2[idx + x_size * y_size];
                    //     if (valid[idx + x_size * y_size]) {
                    //         ++numNeighb;
                    //     }

                    //     sum += vals1[idx - x_size * y_size] + beta * vals2[idx - x_size * y_size];
                    //     if (valid[idx - x_size * y_size]) {
                    //         ++numNeighb;
                    //     }
                    // }
                    out2(cell, c) = (-sum + static_cast<Real>(numNeighb) * center) * invh2;
                    // #if defined(NEON_PLACE_CUDA_DEVICE)
                        atomicAdd(scal, out2(cell, c));
                    // #else
                    //     ;
                    // #endif
                    
                }
            // }
        };
    });
    return cont;
}

// Template instantiations
template class FusedLaplacianMatVec<Neon::domain::eGrid, double>;
template class FusedLaplacianMatVec<Neon::domain::eGrid, float>;
template class FusedLaplacianMatVec<Neon::domain::dGrid, double>;
template class FusedLaplacianMatVec<Neon::domain::dGrid, float>;
// bGrid doesn't support template neighbours
// template class FusedLaplacianMatVec<Neon::domain::bGrid, double>;
// template class FusedLaplacianMatVec<Neon::domain::bGrid, float>;

}  // namespace solver
}  // namespace Neon