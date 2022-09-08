//
// Created by mohammad on 8/18/22.
//
//
// Created by mohammad on 8/18/22.
//
#include "stencil_map.h"

template <typename T>
inline T map_function(T field_val, T other_val)
{
    return ((int) other_val) % 2 ? field_val * other_val : field_val / other_val;
}


template <typename FieldT>
auto mapContainerDGrid(FieldT&  pixels,
                                         int32_t  time) -> Neon::set::Container
{
    return pixels.getGrid().getContainer(
        "MapOperation", [&, time](Neon::set::Loader& L) {
            auto& px = L.load(pixels);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename FieldT::Cell& idx) mutable {
                px(idx, 0) = map_function<double>(px(idx, 0), (double) time);
            };
        });
}


template auto mapContainerDGrid<Neon::domain::dGrid::Field<double, 0>>(Neon::domain::dGrid::Field<double, 0>& pixels, int32_t time) -> Neon::set::Container;
template auto mapContainerDGrid<Neon::domain::dGrid::Field<float, 0>>(Neon::domain::dGrid::Field<float, 0>& pixels, int32_t time) -> Neon::set::Container;


template <typename FieldT>
auto mapContainerDGridFused(FieldT&  pixels,
                       int32_t  time, const int32_t fusion_factor) -> Neon::set::Container
{
    return pixels.getGrid().getContainer(
        "FusedMapBlock", [&, time, fusion_factor](Neon::set::Loader& L) {
            auto& px = L.load(pixels);
            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename FieldT::Cell& idx) mutable {
                for(int i = 0; i < fusion_factor; i++)
                {
                    px(idx, 0) = map_function<double>(px(idx, 0), (double) time);
                }
            };
        });
}


template auto mapContainerDGridFused<Neon::domain::dGrid::Field<double, 0>>(Neon::domain::dGrid::Field<double, 0>& pixels, int32_t time, const int32_t fusion_factor) -> Neon::set::Container;
template auto mapContainerDGridFused<Neon::domain::dGrid::Field<float, 0>>(Neon::domain::dGrid::Field<float, 0>& pixels, int32_t time, const int32_t fusion_factor) -> Neon::set::Container;


template <typename FieldT>
auto stencilContainer(FieldT&  pixels, int32_t num_neighbours) -> Neon::set::Container
{
    return pixels.getGrid().getContainer(
        "StencilOperation", [&, num_neighbours](Neon::set::Loader& L) {
            auto& px = L.load(pixels);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename FieldT::Cell& idx) mutable {
                double result = 0.0;
                for(int i = 0; i < num_neighbours; i++)
                {
//                    auto [val, is_valid] = pixels.nghVal(idx, i, 0, 0);
                    bool is_valid = true;
                    double val = 0.2;
                    result += is_valid ? val : 0.0;
                }
                result /= num_neighbours;
                px(idx, 0) = result;
            };
        });
}


template auto stencilContainer<Neon::domain::dGrid::Field<double, 0>>(Neon::domain::dGrid::Field<double, 0>& pixels, int32_t num_neighbours) -> Neon::set::Container;
template auto stencilContainer<Neon::domain::dGrid::Field<float, 0>>(Neon::domain::dGrid::Field<float, 0>& pixels, int32_t num_neighbours) -> Neon::set::Container;