//
// Created by mohammad on 8/18/22.
//
//
// Created by mohammad on 8/18/22.
//
#include "map.h"

template <typename T, int flop_cnt, int mem_access_cnt>
inline NEON_CUDA_HOST_DEVICE T map_function(T field_val, T other_vals[])
{
    constexpr int per_element_flop_cnt = MAX(int (flop_cnt / mem_access_cnt), 1);
    #pragma unroll
    for(int mem_access = 0; mem_access < mem_access_cnt; mem_access++)
    {
        T other_val = other_vals[mem_access];
        #pragma unroll
        for(int flop = 0; flop < per_element_flop_cnt; flop++)
        {
            field_val = field_val * field_val + other_val;
        }
    }
    return field_val;
}


template <typename FieldT, typename T, int flop_cnt, int mem_access_cnt>
auto mapContainer(FieldT&  pixels,
                                         T  other_vals[]) -> Neon::set::Container
{
    T* other_vals_device;
    cudaMalloc((void **) &other_vals_device, mem_access_cnt * sizeof(T));
    cudaMemcpy(other_vals_device, other_vals, mem_access_cnt * sizeof(T), cudaMemcpyHostToDevice);
    return pixels.getGrid().getContainer(
        "MapOperation", [&, other_vals_device](Neon::set::Loader& L) {
            auto& px = L.load(pixels);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename FieldT::Cell& idx) mutable {
                px(idx, 0) = map_function<T, flop_cnt, mem_access_cnt>(px(idx, 0), other_vals_device);
            };
        });
}


template <typename FieldT, typename T, int fusion_factor, int flop_cnt, int mem_access_cnt>
auto mapContainerFused(FieldT&  pixels,
                       T  other_vals[]) -> Neon::set::Container
{
    T* other_vals_device;
    cudaMalloc((void **) &other_vals_device, mem_access_cnt * sizeof(T));
    cudaMemcpy(other_vals_device, other_vals, mem_access_cnt * sizeof(T), cudaMemcpyHostToDevice);
    return pixels.getGrid().getContainer(
        "FusedMapBlock", [&, other_vals_device](Neon::set::Loader& L) {
            auto& px = L.load(pixels);
            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename FieldT::Cell& idx) mutable {
                #pragma unroll
                for(int i = 0; i < fusion_factor; i++)
                {
                    px(idx, 0) = map_function<T, flop_cnt, mem_access_cnt>(px(idx, 0), other_vals_device);
                }
            };
        });
}



template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;


template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;