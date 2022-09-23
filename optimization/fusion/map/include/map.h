//
// Created by mohammad on 8/18/22.
//
#pragma once

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/Neon.h"
#include "Neon/skeleton/Skeleton.h"


#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define FLOP_CNT 1
#define MEM_ACCESS_CNT 1


template <typename T>
inline T map_function(T field_val, T other_val);


template <typename FieldT, typename T, int flop_cnt, int mem_access_cnt>
auto mapContainer(FieldT&  pixels,
                                         T other_vals[])  -> Neon::set::Container;


template <typename FieldT, typename T, int fusion_factor, int flop_cnt, int mem_access_cnt>
auto mapContainerFused(FieldT&  pixels,
                            T other_vals[]) -> Neon::set::Container;


// template <typename T>
// std::vector<Neon::set::Container> make_fused_container(Neon::domain::dGrid::Field<T, 0> field_fused, T other_vals[], const int fusion_factor, const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS);


// template <typename T>
// std::vector<Neon::set::Container> make_baseline_container(Neon::domain::dGrid::Field<T, 0> field_baseline,T other_vals[], const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS);

extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 1, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 2, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 4, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 8, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 16, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 32, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 64, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 128, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 128, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 128, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 128, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 256, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 256, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 256, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 256, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 512, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 512, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 512, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<double, 0>, double, 512, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 1, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 2, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 4, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 8, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 16, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 32, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 64, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 128, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 128, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 128, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 128, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 256, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 256, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 256, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 256, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 512, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 512, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 512, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainerFused<Neon::domain::dGrid::Field<float, 0>, float, 512, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;


extern template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 1, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 2, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 4, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<double, 0>, double, 8, 1>(Neon::domain::dGrid::Field<double, 0>& pixels, double other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 1, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 2, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 4, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;
extern template auto mapContainer<Neon::domain::dGrid::Field<float, 0>, float, 8, 1>(Neon::domain::dGrid::Field<float, 0>& pixels, float other_vals[]) -> Neon::set::Container;