if __name__ == "__main__":
    fusion_factors = [1, 2, 4, 8, 16, 32, 64]
    flop_cnts = [1, 2, 4, 8]
    mem_access_cnts = [1]


    function_calls = ['#include "map.h"\n\n\n',
                        "template <typename T>\n"
                        "std::vector<Neon::set::Container> make_fused_container(Neon::domain::dGrid::Field<T, 0> field_fused, T other_vals[], const int fusion_factor, const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS)\n",
                        "{\n",
                        "\tstd::vector<Neon::set::Container> container_fused;\n",
                        "\tfor(int block = 0; block < NUM_BLOCKS; block++)\n",
                        "\t{\n"]
    for FUSION_FACTOR in fusion_factors:
        for FLOP_CNT in flop_cnts:
            for MEM_ACCESS_CNT in mem_access_cnts:
                if_statement = "if" if FUSION_FACTOR == 1 and FLOP_CNT == 1 and MEM_ACCESS_CNT == 1 else "else if"
                function_calls.append("\t\t" + if_statement + f"(fusion_factor == {FUSION_FACTOR} && flop_cnt == {FLOP_CNT} && mem_access_cnt == {MEM_ACCESS_CNT})\n" + 
                    f"\t\t\tcontainer_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, {FUSION_FACTOR}, {FLOP_CNT}, {MEM_ACCESS_CNT}>(field_fused, other_vals));\n")
    function_calls += ["\t}\n",
                        "\treturn container_fused;\n",
                        "}\n\n"]
    function_calls += ["template <typename T>\n",
                        "std::vector<Neon::set::Container> make_baseline_container(Neon::domain::dGrid::Field<T, 0> field_baseline,T other_vals[], const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS)\n",
                        "{\n",
                        "\tstd::vector<Neon::set::Container> container_baseline;\n",
                        "\tfor(int block = 0; block < NUM_BLOCKS; block++)\n",
                        "\t{\n"]
    for FLOP_CNT in flop_cnts:
        for MEM_ACCESS_CNT in mem_access_cnts:
            if_statement = "if" if FLOP_CNT == 1 and MEM_ACCESS_CNT == 1 else "else if"
            function_calls.append("\t\t" + if_statement + f"(flop_cnt == {FLOP_CNT} && mem_access_cnt == {MEM_ACCESS_CNT})\n" + 
                f"\t\t\tcontainer_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, {FLOP_CNT}, {MEM_ACCESS_CNT}>(field_baseline, other_vals));\n")
    
    function_calls += ["\t}\n",
                        "\treturn container_baseline;\n",
                        "}\n"]
    with open("./optimization/fusion/map/src/function_calls.cpp", "w") as file:
        file.writelines(function_calls)