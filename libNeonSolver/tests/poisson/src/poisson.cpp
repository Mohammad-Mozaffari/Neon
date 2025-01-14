#include "Poisson.h"

// Specialization for eGrid_t
template <>
Neon::domain::eGrid createGrid<Neon::domain::eGrid>(const Neon::Backend& backend, int domainSize)
{
    using namespace Neon;
    using Neon::domain::eGrid;

    // Create a dense grid
    index_3d                      cellDomain(domainSize, domainSize, domainSize);
    std::function<bool(index_3d)> activeCells = [](index_3d /*target*/) -> bool {
        return true;
    };

    // Create a 6-neighbor stencil for Laplacian kernel
    eGrid grid(backend, cellDomain, activeCells, Neon::domain::Stencil::s7_Laplace_t());
    return grid;
}


// Specialization for dGrid_t
template <>
Neon::domain::dGrid createGrid<Neon::domain::dGrid>(const Neon::Backend& backend, int domainSize)
{
    using namespace Neon;
    using Neon::domain::dGrid;

    // Create a dense grid
    index_3d                      cellDomain(domainSize, domainSize, domainSize);
    std::function<bool(index_3d)> activeCells = [](index_3d /*target*/) -> bool {
        return true;
    };

    // Create a 6-neighbor stencil for Laplacian kernel
    dGrid grid(backend, cellDomain, activeCells, Neon::domain::Stencil::s7_Laplace_t());
    return grid;
}

// bGrid does not support swap
// Specialization for bGrid_t
// template <>
// Neon::domain::bGrid createGrid<Neon::domain::bGrid>(const Neon::Backend& backend, int domainSize)
// {
//     using namespace Neon;
//     using Neon::domain::bGrid;

//     // Create a dense grid
//     index_3d                      cellDomain(domainSize, domainSize, domainSize);
//     std::function<bool(index_3d)> activeCells = [](index_3d /*target*/) -> bool {
//         return true;
//     };

//     // Create a 6-neighbor stencil for Laplacian kernel
//     bGrid grid(backend, cellDomain, activeCells, Neon::domain::Stencil::s7_Laplace_t());
//     return grid;
// }

#define TEMPLATE_INST(GRID, REAL, CARD)                                                                 \
    template auto testPoissonContainers<GRID, REAL, CARD>(const Neon::Backend&    backend,       \
                                                          const std::string&             solverName,    \
                                                          int                            domainSize,    \
                                                          std::array<REAL, CARD>         bdZmin,        \
                                                          std::array<REAL, CARD>         bdZmax,        \
                                                          size_t                         maxIterations, \
                                                          REAL                           tolerance,     \
                                                          Neon::skeleton::Occ occE,          \
                                                          Neon::set::TransferMode transferE)     \
        ->std::pair<Neon::solver::SolverResultInfo,                                                     \
                    Neon::solver::SolverStatus>;

TEMPLATE_INST(Neon::domain::dGrid, double, 1)
TEMPLATE_INST(Neon::domain::dGrid, double, 3)
// bGrid does not support swap
// TEMPLATE_INST(Neon::domain::bGrid, double, 1)
// TEMPLATE_INST(Neon::domain::bGrid, double, 3)
TEMPLATE_INST(Neon::domain::eGrid, double, 1)
TEMPLATE_INST(Neon::domain::eGrid, double, 3)

TEMPLATE_INST(Neon::domain::dGrid, float, 1)
TEMPLATE_INST(Neon::domain::dGrid, float, 3)
// bGrid does not support swap
// TEMPLATE_INST(Neon::domain::bGrid, float, 1)
// TEMPLATE_INST(Neon::domain::bGrid, float, 3)
TEMPLATE_INST(Neon::domain::eGrid, float, 1)
TEMPLATE_INST(Neon::domain::eGrid, float, 3)

#undef TEMPLATE_INST