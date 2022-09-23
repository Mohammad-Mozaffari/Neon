//
// Created by mohammad on 8/17/22.
//

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
#include "gtest/gtest.h"

#include "function_calls.cpp"
#include "map.h"


#define SPEEDUP 0
#define BASELINE 1
#define FUSED 2

using namespace Neon::set;
using namespace Neon::domain;

std::vector<int>        DEVICES;            // GPU device IDs
int                     DOMAIN_SIZE = 256;  // Number of voxels along each axis
size_t                  MAX_ITER = 10;      // Maximum iterations for the solver
double                  TOL = 1e-10;        // Absolute tolerance for use in converge check
int                     CARDINALITY = 1;
std::string             GRID_TYPE = "dGrid";
std::string             DATA_TYPE = "double";
std::string             REPORT_FILENAME = "fusion_map_report";
int                     WARMUP = 0;
int                     TIMES = 1;
int                     NUM_BLOCKS = 1;
int                     FUSION_FACTOR = 1;
int                     EVAL_SCALABILITY = 0;
int                     RUN_BASELINE = 1;
Neon::skeleton::Occ     occE = Neon::skeleton::Occ::none;
Neon::set::TransferMode transferE = Neon::set::TransferMode::get;
int                     ARGC;
char**                  ARGV;


template <typename T>
std::vector<Neon::set::Container> make_fused_container(Neon::domain::dGrid::Field<T, 0> field_fused, T other_vals[], const int fusion_factor, const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS)
{
    std::vector<Neon::set::Container> container_fused;
    for (int block = 0; block < NUM_BLOCKS; block++) {
        if (fusion_factor == 1 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 1, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 1 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 1, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 1 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 1, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 1 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 1, 8, 1>(field_fused, other_vals));
        else if (fusion_factor == 2 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 2, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 2 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 2, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 2 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 2, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 2 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 2, 8, 1>(field_fused, other_vals));
        else if (fusion_factor == 4 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 4, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 4 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 4, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 4 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 4, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 4 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 4, 8, 1>(field_fused, other_vals));
        else if (fusion_factor == 8 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 8, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 8 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 8, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 8 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 8, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 8 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 8, 8, 1>(field_fused, other_vals));
        else if (fusion_factor == 16 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 16, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 16 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 16, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 16 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 16, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 16 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 16, 8, 1>(field_fused, other_vals));
        else if (fusion_factor == 32 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 32, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 32 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 32, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 32 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 32, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 32 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 32, 8, 1>(field_fused, other_vals));
        else if (fusion_factor == 64 && flop_cnt == 1 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 64, 1, 1>(field_fused, other_vals));
        else if (fusion_factor == 64 && flop_cnt == 2 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 64, 2, 1>(field_fused, other_vals));
        else if (fusion_factor == 64 && flop_cnt == 4 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 64, 4, 1>(field_fused, other_vals));
        else if (fusion_factor == 64 && flop_cnt == 8 && mem_access_cnt == 1)
            container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 64, 8, 1>(field_fused, other_vals));
    }
    return container_fused;
}

template <typename T>
std::vector<Neon::set::Container> make_baseline_container(Neon::domain::dGrid::Field<T, 0> field_baseline, T other_vals[], const int flop_cnt, const int mem_access_cnt, const int NUM_BLOCKS)
{
    std::vector<Neon::set::Container> container_baseline;
    for (int block = 0; block < NUM_BLOCKS; block++) {
        if (flop_cnt == 1 && mem_access_cnt == 1)
            container_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, 1, 1>(field_baseline, other_vals));
        else if (flop_cnt == 2 && mem_access_cnt == 1)
            container_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, 2, 1>(field_baseline, other_vals));
        else if (flop_cnt == 4 && mem_access_cnt == 1)
            container_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, 4, 1>(field_baseline, other_vals));
        else if (flop_cnt == 8 && mem_access_cnt == 1)
            container_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, 8, 1>(field_baseline, other_vals));
    }
    return container_baseline;
}


template <typename T>
std::vector<double> mapFusionPerfTest(bool generate_report = true, std::vector<int> devices = {0}, int domain_size = 1, int fusion_factor = 1, int flop_cnt = 1, int mem_access_cnt = 1, int max_num_blocks = 1, bool test_baseline = true)
{
    assert(GRID_TYPE == "dGrid" || GRID_TYPE == "eGrid" || GRID_TYPE == "bGrid");
    assert(DATA_TYPE == "double" || DATA_TYPE == "single");

    int num_blocks = MAX(max_num_blocks / fusion_factor / MAX(flop_cnt, mem_access_cnt), 1);


    if (devices.empty()) {
        devices.push_back(0);
    }
    DevSet        deviceSet(Neon::DeviceType::CUDA, devices);
    Neon::Backend backend(deviceSet, Neon::Runtime::stream);
    backend.setAvailableStreamSet(2);


    // Run on different domain sizes
    std::vector<double> experimentTime(TIMES);
    std::vector<double> totalTime(TIMES);


    Neon::Timer_ms timer_total, timer_computation_baseline, timer_computation_fused;
    timer_total.start();


    Neon::domain::Stencil emptyStencil([] {
        return std::vector<Neon::index_3d>{
            {1, 0, 0},
            {0, 1, 0}};
    }());

    Neon::index_3d dom(domain_size, domain_size, domain_size);

    using Grid = Neon::domain::dGrid;
    Grid grid(
        backend,
        dom,
        [&](const Neon::index_3d&) -> bool {
            return true;
        },
        emptyStencil);


    auto field_baseline = grid.newField<T>("field_baseline",
                                           1,
                                           -100);
    auto field_fused = grid.newField<T>("field_fused",
                                        1,
                                        -100);

    field_baseline.forEachActiveCell([&](const Neon::index_3d&, const int&, T& value) {
        value = 1.0;
    });
    field_fused.forEachActiveCell([&](const Neon::index_3d&, const int&, T& value) {
        value = 1.0;
    });


    Neon::skeleton::Skeleton skeleton_baseline(backend), skeleton_fused(backend);

    int32_t time = 0;
    T       other_vals[MEM_ACCESS_CNT];
    for (int i = 0; i < MEM_ACCESS_CNT; i++) {
        other_vals[i] = i + 1;
    }

    std::vector<Neon::set::Container> container_baseline = make_baseline_container<T>(field_baseline, other_vals, flop_cnt, mem_access_cnt, num_blocks * fusion_factor);
    std::vector<Neon::set::Container> container_fused = make_fused_container<T>(field_fused, other_vals, fusion_factor, flop_cnt, mem_access_cnt, num_blocks);

    skeleton_baseline.sequence(container_baseline, "map_baseline");
    skeleton_fused.sequence(container_fused, "map_fused");
    skeleton_baseline.ioToDot("graphs/map_fusion_baseline_GPUCount:" + std::to_string(devices.size()) +
                              "_MemoryAccessCount:" + std::to_string(mem_access_cnt) +
                              "_FlopCount:" + std::to_string(flop_cnt) +
                              "_FusionFactor:" + std::to_string(fusion_factor) +
                              "_DomainSize:" + std::to_string(domain_size));
    skeleton_fused.ioToDot("graphs/map_fusion_fused_GPUCount:" + std::to_string(devices.size()) +
                           "_MemoryAccessCount:" + std::to_string(mem_access_cnt) +
                           "_FlopCount:" + std::to_string(flop_cnt) +
                           "_FusionFactor:" + std::to_string(fusion_factor) +
                           "_DomainSize:" + std::to_string(domain_size));

    std::cout << "TEST_BASELINE" << test_baseline << std::endl;
    

    for (time = 0; time < TIMES + WARMUP; ++time) {
        if (time == WARMUP) {
            timer_computation_baseline.start();
        }
        if (test_baseline) {
            skeleton_baseline.run();
        }
    }
    timer_computation_baseline.stop();
    if (test_baseline) {
        field_baseline.updateIO(0);
    }


    for (time = 0; time < TIMES + WARMUP; ++time) {
        if (time == WARMUP) {
            timer_computation_fused.start();
        }
        skeleton_fused.run();
    }
    timer_computation_fused.stop();
    field_fused.updateIO(0);

    timer_total.stop();
    double speedup = timer_computation_baseline.time() / timer_computation_fused.time();
    if (generate_report) {
        // Create a report
        Neon::Report report("MapFusion_" + std::string(GRID_TYPE) + "_" + std::to_string(CARDINALITY) + "D_" + std::to_string(DEVICES.size()) + "GPUs");

        // report.setToken("Token404");

        report.commandLine(ARGC, ARGV);

        report.addMember("voxelDomain", dom.to_stringForComposedNames());
        report.addMember("cardinality", CARDINALITY);
        report.addMember("numGPUs", DEVICES.size());
        report.addMember("gridType", GRID_TYPE);
        report.addMember("dataType", DATA_TYPE);
        report.addMember("skeletonOCC", Neon::skeleton::OccUtils::toString(occE));
        report.addMember("skeletonTransferMode", Neon::set::TransferModeUtils::toString(transferE));
        report.addMember("TimeTotal_ms", timer_total.time());
        report.addMember("TimeComputationBaseline_ms", timer_computation_baseline.time());
        report.addMember("TimeComputationFused_ms", timer_computation_fused.time());
        report.addMember("FusionFactor", FUSION_FACTOR);
        report.addMember("Speedup", speedup);

        std::stringstream stringstream;
        stringstream << "Saving report file here: " << REPORT_FILENAME << std::endl;
        NEON_INFO(stringstream.str());

        report.write(REPORT_FILENAME);
    }

    std::vector<double> result = {speedup, timer_computation_baseline.time(), timer_computation_fused.time()};

    return result;
}


template <typename T>
std::vector<T> log_space(T max, T min = 1, T base = 2)
{
    std::vector<T> result;
    for (T val = min; val <= max; val *= base) {
        result.push_back(val);
    }
    return result;
}


template <typename T>
std::vector<T> lin_space(T max, T min = 1, T base = 1)
{
    std::vector<T> result;
    for (T val = min; val <= max; val += base) {
        result.push_back(val);
    }
    return result;
}


int main(int argc, char** argv)
{
    ARGC = argc;
    ARGV = argv;

    Neon::init();

    // CLI for performance test
    auto cli =
        (clipp::option("--gpus") & clipp::integers("gpus", DEVICES) % "GPU ids to use",
         clipp::option("--grid") & clipp::value("grid", GRID_TYPE) % "Could be eGrid, dGrid, or bGrid",
         clipp::option("--data_type") & clipp::value("data_type", DATA_TYPE) % "Could be single or double",
         clipp::option("--cardinality") & clipp::value("cardinality", CARDINALITY) % "Must be 1 or 3",
         clipp::option("--domain_size") & clipp::integer("domain_size", DOMAIN_SIZE) % "Voxels along each dimension of the cube domain",
         clipp::option("--report_filename ") & clipp::value("report_filename", REPORT_FILENAME) % "Output report filename",
         clipp::option("--times ") & clipp::integer("times", TIMES) % "Times to run the experiment",
         clipp::option("--warmup ") & clipp::integer("warmup", WARMUP) % "Times to run the experiment for warmup",
         clipp::option("--fusion_factor") & clipp::integer("fusion_factor", FUSION_FACTOR) % "Number of map operations to be fused",
         clipp::option("--num_blocks") & clipp::integer("num_blocks", NUM_BLOCKS) % "Number of fused map operation blocks",
         clipp::option("--eval_scalability") & clipp::integer("eval_scalability", EVAL_SCALABILITY) % "Evaluate scalability",
         clipp::option("--run_baseline") & clipp::integer("run_baseline", RUN_BASELINE) % "Run baseline and fused version",
         ((clipp::option("--sOCC ").set(occE, Neon::skeleton::Occ::standard) % "Standard OCC") |
          (clipp::option("--nOCC ").set(occE, Neon::skeleton::Occ::none) % "No OCC (on by default)") |
          (clipp::option("--eOCC ").set(occE, Neon::skeleton::Occ::extended) % "Extended OCC") |
          (clipp::option("--e2OCC ").set(occE, Neon::skeleton::Occ::twoWayExtended) % "Two-way Extended OCC")),
         ((clipp::option("--put ").set(transferE, Neon::set::TransferMode::put) % "Set transfer mode to GET") |
          (clipp::option("--get ").set(transferE, Neon::set::TransferMode::get) % "Set transfer mode to PUT (on by default)")));


    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
        return -1;
    }
    std::cout << " #gpus= " << (DEVICES.empty() ? 1 : DEVICES.size()) << "\n";
    std::cout << " grid= " << GRID_TYPE << "\n";
    std::cout << " data_type= " << DATA_TYPE << "\n";
    std::cout << " cardinality= " << CARDINALITY << "\n";
    std::cout << " domain_size= " << DOMAIN_SIZE << "\n";
    std::cout << " times= " << TIMES << "\n";
    std::cout << " fusion_factor= " << FUSION_FACTOR << "\n";
    std::cout << " OCC= " << Neon::skeleton::OccUtils::toString(occE) << "\n";
    std::cout << " transfer= " << Neon::set::TransferModeUtils::toString(transferE) << "\n";

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        if (EVAL_SCALABILITY) {
            Neon::Report report("MapFusion_" + std::string(GRID_TYPE) + "_" + std::to_string(CARDINALITY) + "D_" + std::to_string(DEVICES.size()) + "GPUs");

            report.addMember("cardinality", CARDINALITY);
            report.addMember("numGPUs", DEVICES.size());
            report.addMember("gridType", GRID_TYPE);
            report.addMember("dataType", DATA_TYPE);
            report.addMember("skeletonOCC", Neon::skeleton::OccUtils::toString(occE));
            report.addMember("skeletonTransferMode", Neon::set::TransferModeUtils::toString(transferE));
            std::vector<int> domain_sizes = log_space<int>(DOMAIN_SIZE, 32), gpu_counts = lin_space<int>(DEVICES.size());
            int              max_num_blocks = NUM_BLOCKS;
            auto             all_devices = DEVICES;
            std::vector<int> devices;


            std::vector<int> fusion_factors = log_space<int>(FUSION_FACTOR);
            std::vector<int> flop_cnts = log_space<int>(8);
            std::vector<int> mem_acess_cnts = {1};
            devices.clear();
            for (int gpu_count : gpu_counts) {
                devices.push_back(all_devices[gpu_count - 1]);
                for (int fusion_factor : fusion_factors) {
                    for (int flop_cnt : flop_cnts) {
                        for (int mem_access_cnt : mem_acess_cnts) {
                            std::vector<double> experiment_results;
                            for (int domain_size : domain_sizes) {
                                if (DATA_TYPE == "single") {
                                    experiment_results = mapFusionPerfTest<float>(false, devices = devices, domain_size = domain_size, fusion_factor = fusion_factor, flop_cnt = flop_cnt, mem_access_cnt = mem_access_cnt, max_num_blocks = max_num_blocks);
                                } else if (DATA_TYPE == "double") {
                                    experiment_results = mapFusionPerfTest<double>(false, devices = devices, domain_size = domain_size, fusion_factor = fusion_factor, flop_cnt = flop_cnt, mem_access_cnt = mem_access_cnt, max_num_blocks = max_num_blocks);
                                } else {
                                    return -1;
                                }
                                report.addMember("GPUCount:" + std::to_string(gpu_count) +
                                                     "_MemoryAccessCount:" + std::to_string(mem_access_cnt) +
                                                     "_FlopCount:" + std::to_string(flop_cnt) +
                                                     "_FusionFactor:" + std::to_string(fusion_factor) +
                                                     "_DomainSize:" + std::to_string(domain_size),
                                                 experiment_results);
                            }
                        }
                    }
                }
                std::stringstream stringstream;
                stringstream << "Saving report file here: " << REPORT_FILENAME << std::endl;
                NEON_INFO(stringstream.str());
            }

            report.write(REPORT_FILENAME);
        } else {
            std::vector<double> experiment_results;
            if (DATA_TYPE == "single") {
                experiment_results = mapFusionPerfTest<float>(false, DEVICES, DOMAIN_SIZE, FUSION_FACTOR, 1, 1, NUM_BLOCKS, RUN_BASELINE);
            } else if (DATA_TYPE == "double") {
                experiment_results = mapFusionPerfTest<double>(false, DEVICES, DOMAIN_SIZE, FUSION_FACTOR, 1, 1, NUM_BLOCKS, RUN_BASELINE);
            } else {
                return -1;
            }
            std::cout << "Speedup: " << experiment_results[0] << std::endl;
            std::cout << "Baseline Time: " << experiment_results[1] << std::endl;
            std::cout << "Fused Time: " << experiment_results[2] << std::endl;
        }
    }
    return 0;
}
