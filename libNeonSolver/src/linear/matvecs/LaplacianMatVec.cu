#include "Neon/core/core.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/solver/linear/matvecs/LaplacianMatVec.h"

namespace Neon {
namespace solver {

template <typename Grid, typename Real>
inline Neon::set::Container LaplacianMatVec<Grid, Real>::matVec(const Field&   input,
                                                                      const bdField& boundary,
                                                                      Field&         output)
{
    Real stepSize = m_h;

    auto cont = input.getGrid().getContainer("Laplacian", [&, stepSize](Neon::set::Loader& L) {
        auto& inp = L.load(input, Neon::Compute::STENCIL);
        auto& bnd = L.load(boundary);
        auto& out = L.load(output);

        // Precompute 1/h^2
        const Real invh2 = Real(1.0) / (stepSize * stepSize);

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& cell) mutable {
            const int cardinality = inp.cardinality();

            // Iterate through each element's cardinality
            for (int c = 0; c < cardinality; ++c) {
                const Real center = inp(cell, c);
                if (bnd(cell, c) == 0) {
                    out(cell, c) = center;
                } else {
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
                    } else {
                        typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        ngh.x = 1;
                        ngh.y = 0;
                        ngh.z = 0;
                        auto neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-x
                        ngh.x = -1;
                        ngh.y = 0;
                        ngh.z = 0;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //+y
                        ngh.x = 0;
                        ngh.y = 1;
                        ngh.z = 0;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-y
                        ngh.x = 0;
                        ngh.y = -1;
                        ngh.z = 0;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //+z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = 1;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);

                        //-z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = -1;
                        neighbor = inp.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor);
                    }
                    out(cell, c) = (-sum + static_cast<Real>(numNeighb) * center) * invh2;
                }
            }
        };
    });
    return cont;
}

// Template instantiations
template class LaplacianMatVec<Neon::domain::eGrid, double>;
template class LaplacianMatVec<Neon::domain::eGrid, float>;
template class LaplacianMatVec<Neon::domain::dGrid, double>;
template class LaplacianMatVec<Neon::domain::dGrid, float>;
template class LaplacianMatVec<Neon::domain::bGrid, double>;
template class LaplacianMatVec<Neon::domain::bGrid, float>;

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
                                                                      Field&         output2)
{
    Real stepSize = this->stepSize();

    auto cont = input1.getGrid().getContainer("Laplacian", [&, stepSize](Neon::set::Loader& L) {
        auto& inp1 = L.load(input1, Neon::Compute::STENCIL);
        auto& inp2 = L.load(input2, Neon::Compute::STENCIL);
        auto& bnd = L.load(boundary);
        auto& out1 = L.load(output1);
        auto& out2 = L.load(output2);

        Real beta;
        if (std::abs(delta_old) > std::numeric_limits<Real>::epsilon()) {
            beta = delta_new / delta_old;
        } else {
            beta = 0;
        }

        // Precompute 1/h^2
        const Real invh2 = Real(1.0) / (stepSize * stepSize);

        return [=] NEON_CUDA_HOST_DEVICE(const typename Grid::template Field<Real>::Cell& cell) mutable {
            const int cardinality = inp1.cardinality();

            // Iterate through each element's cardinality
            for (int c = 0; c < cardinality; ++c) {
                const Real center1 = inp1(cell, c);
                const Real center2 = inp2(cell, c);

                const Real center = center1 + beta * center2;
                out1(cell, c) = center;
                if (bnd(cell, c) == 0) {
                    out1(cell, c) = center1; //DOUBLE CHECK
                    out2(cell, c) = center2;
                } else {
                    Real       sum(0.0);
                    int           numNeighb = 0;
                    const Real defaultVal{0};

                    auto checkNeighbor = [&sum, &numNeighb, &beta](Neon::domain::NghInfo<Real>& neighbor1, Neon::domain::NghInfo<Real>& neighbor2) {
                        if (neighbor1.isValid && neighbor2.isValid) {
                            ++numNeighb;
                            sum += neighbor1.value + beta * neighbor2.value;
                        }
                    };
                    // Laplacian stencil operates on 6 neighbors (assuming 3D)
                    if constexpr (std::is_same<Grid, Neon::domain::internal::eGrid::eGrid>::value) {
                        for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                            auto neighbor1 = inp1.nghVal(cell, nghIdx, c, defaultVal);
                            auto neighbor2 = inp2.nghVal(cell, nghIdx, c, defaultVal);
                            checkNeighbor(neighbor1, neighbor2);
                        }
                    } else {
                        typename Grid::template Field<Real, 0>::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        ngh.x = 1;
                        ngh.y = 0;
                        ngh.z = 0;
                        auto neighbor1 = inp1.nghVal(cell, ngh, c, defaultVal);
                        auto neighbor2 = inp2.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor1, neighbor2);

                        //-x
                        ngh.x = -1;
                        ngh.y = 0;
                        ngh.z = 0;
                        neighbor1 = inp1.nghVal(cell, ngh, c, defaultVal);
                        neighbor2 = inp2.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor1, neighbor2);

                        //+y
                        ngh.x = 0;
                        ngh.y = 1;
                        ngh.z = 0;
                        neighbor1 = inp1.nghVal(cell, ngh, c, defaultVal);
                        neighbor2 = inp2.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor1, neighbor2);

                        //-y
                        ngh.x = 0;
                        ngh.y = -1;
                        ngh.z = 0;
                        neighbor1 = inp1.nghVal(cell, ngh, c, defaultVal);
                        neighbor2 = inp2.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor1, neighbor2);

                        //+z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = 1;
                        neighbor1 = inp1.nghVal(cell, ngh, c, defaultVal);
                        neighbor2 = inp2.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor1, neighbor2);

                        //-z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = -1;
                        neighbor1 = inp1.nghVal(cell, ngh, c, defaultVal);
                        neighbor2 = inp2.nghVal(cell, ngh, c, defaultVal);
                        checkNeighbor(neighbor1, neighbor2);
                    }
                    out2(cell, c) = (-sum + static_cast<Real>(numNeighb) * center) * invh2;
                }
            }
        };
    });
    return cont;
}

// Template instantiations
template class FusedLaplacianMatVec<Neon::domain::eGrid, double>;
template class FusedLaplacianMatVec<Neon::domain::eGrid, float>;
template class FusedLaplacianMatVec<Neon::domain::dGrid, double>;
template class FusedLaplacianMatVec<Neon::domain::dGrid, float>;
template class FusedLaplacianMatVec<Neon::domain::bGrid, double>;
template class FusedLaplacianMatVec<Neon::domain::bGrid, float>;

}  // namespace solver
}  // namespace Neon