//
// Created by mohammad on 8/18/22.
//
#pragma once

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/Neon.h"
#include "Neon/skeleton/Skeleton.h"


template <typename T>
inline T map_function(T field_val, T other_val);


template <typename FieldT>
auto mapContainerDGrid(FieldT&  pixels, int32_t time)  -> Neon::set::Container;


extern template auto mapContainerDGrid<Neon::domain::dGrid::Field<double, 0>>(Neon::domain::dGrid::Field<double, 0>& pixels, int32_t time) -> Neon::set::Container;
extern template auto mapContainerDGrid<Neon::domain::dGrid::Field<float, 0>>(Neon::domain::dGrid::Field<float, 0>& pixels, int32_t time) -> Neon::set::Container;


template <typename FieldT>
auto mapContainerDGridFused(FieldT&  pixels,
                            int32_t  time, const int32_t fusion_factor) -> Neon::set::Container;


extern template auto mapContainerDGridFused<Neon::domain::dGrid::Field<double, 0>>(Neon::domain::dGrid::Field<double, 0>& pixels, int32_t time, const int32_t fusion_factor) -> Neon::set::Container;
extern template auto mapContainerDGridFused<Neon::domain::dGrid::Field<float, 0>>(Neon::domain::dGrid::Field<float, 0>& pixels, int32_t time, const int32_t fusion_factor) -> Neon::set::Container;


template <typename FieldT>
auto stencilContainer(FieldT&  pixels, int32_t num_neighbours) -> Neon::set::Container;


extern template auto stencilContainer<Neon::domain::dGrid::Field<double, 0>>(Neon::domain::dGrid::Field<double, 0>& pixels, int32_t num_neighbours) -> Neon::set::Container;
extern template auto stencilContainer<Neon::domain::dGrid::Field<float, 0>>(Neon::domain::dGrid::Field<float, 0>& pixels, int32_t num_neighbours) -> Neon::set::Container;