#include "map.h"


template <typename T>
std::vector<Neon::set::Container> make_fused_container(Neon::domain::dGrid::Field<T, 0> field_fused, T other_vals[], const int fusion_factor, const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS)
{
	std::vector<Neon::set::Container> container_fused;
	for(int block = 0; block < NUM_BLOCKS; block++)
	{
		if(fusion_factor == 1 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 1, 1, 1>(field_fused, other_vals));
		else if(fusion_factor == 2 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 2, 1, 1>(field_fused, other_vals));
		else if(fusion_factor == 4 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 4, 1, 1>(field_fused, other_vals));
		else if(fusion_factor == 8 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 8, 1, 1>(field_fused, other_vals));
		else if(fusion_factor == 16 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 16, 1, 1>(field_fused, other_vals));
		else if(fusion_factor == 32 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 32, 1, 1>(field_fused, other_vals));
		else if(fusion_factor == 64 && flop_cnt == 1 && mem_access_cnt == 1)
			container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 64, 1, 1>(field_fused, other_vals));
	}
	return container_fused;
}

template <typename T>
std::vector<Neon::set::Container> make_baseline_container(Neon::domain::dGrid::Field<T, 0> field_baseline,T other_vals[], const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS)
{
	std::vector<Neon::set::Container> container_baseline;
	for(int block = 0; block < NUM_BLOCKS; block++)
	{
		if(flop_cnt == 1 && mem_access_cnt == 1)
			container_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, 1, 1>(field_baseline, other_vals));
	}
	return container_baseline;
}
