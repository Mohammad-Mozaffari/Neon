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

#include "map.h"


#define SPEEDUP 0
#define BASELINE 1
#define FUSED 2

using namespace Neon::set;
using namespace Neon::domain;

std::vector<int>             DEVICES;            // GPU device IDs
int                          DOMAIN_SIZE = 256;  // Number of voxels along each axis
size_t                       MAX_ITER = 10;      // Maximum iterations for the solver
double                       TOL = 1e-10;        // Absolute tolerance for use in converge check
int                          CARDINALITY = 1;
std::string                  GRID_TYPE = "dGrid";
std::string                  DATA_TYPE = "double";
std::string                  REPORT_FILENAME = "fusion_map_report";
int                          WARMUP = 0;
int                          TIMES = 1;
int                          NUM_BLOCKS = 1;
int                          FUSION_FACTOR = 1;
int                          EVAL_SCALABILITY = 0;
Neon::skeleton::Occ occE = Neon::skeleton::Occ::none;
Neon::set::TransferMode transferE = Neon::set::TransferMode::get;
int                          ARGC;
char**                       ARGV;


template <typename T>
std::vector<double> mapFusionPerfTest(bool generate_report=true, int fusion_factor=1, int flop_cnt=1, int mem_access_cnt=1)
{
    assert(GRID_TYPE == "dGrid" || GRID_TYPE == "eGrid" || GRID_TYPE == "bGrid");
    assert(DATA_TYPE == "double" || DATA_TYPE == "single");


    if (DEVICES.empty()) {
        DEVICES.push_back(0);
    }
    DevSet        deviceSet(Neon::DeviceType::CUDA, DEVICES);
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

    Neon::index_3d dom(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE);

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
    T other_vals[MEM_ACCESS_CNT];
    for(int i = 0; i < MEM_ACCESS_CNT; i++)
    {
        other_vals[i] = i + 1;
    }

    std::vector<Neon::set::Container> container_baseline = make_baseline_container<T>(field_baseline, other_vals, flop_cnt, mem_access_cnt, NUM_BLOCKS);
    std::vector<Neon::set::Container> container_fused = make_fused_container<T>(field_fused, other_vals, fusion_factor, flop_cnt, mem_access_cnt, NUM_BLOCKS);
    // for(int block = 0; block < NUM_BLOCKS; block++)
    // {
    //     if(FUSION_FACTOR == 1)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 1, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));
    //     else if(FUSION_FACTOR == 2)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T,  2, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));
    //     else if(FUSION_FACTOR == 4)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 4, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));
    //     else if(FUSION_FACTOR == 8)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T,  8, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));
    //     else if(FUSION_FACTOR == 16)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T,  16, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));
    //     else if(FUSION_FACTOR == 32)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T,  32, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));
    //     else if(FUSION_FACTOR == 64)
    //         container_fused.push_back(mapContainerFused<Neon::domain::dGrid::Field<T, 0>, T, 64, FLOP_CNT, MEM_ACCESS_CNT>(field_fused, other_vals));

    //     for(int map_op = 0; map_op < FUSION_FACTOR; map_op++)
    //     {
    //         container_baseline.push_back(mapContainer<Neon::domain::dGrid::Field<T, 0>, T, FLOP_CNT, MEM_ACCESS_CNT>(field_baseline, other_vals));
    //     }
    // }
    skeleton_baseline.sequence(container_baseline, "map_baseline");
    skeleton_fused.sequence(container_fused, "map_fused");
    skeleton_baseline.ioToDot("map_fusion_baseline");
    skeleton_fused.ioToDot("map_fusion_fused");

    for (time = 0; time < TIMES + WARMUP; ++time) {
        if(time == WARMUP)
        {
            timer_computation_baseline.start();
        }
        skeleton_baseline.run();
        field_baseline.updateIO(0);
    }
    timer_computation_baseline.stop();


    for (time = 0; time < TIMES + WARMUP; ++time) {
        if(time == WARMUP)
        {
            timer_computation_fused.start();
        }
        skeleton_fused.run();
        field_fused.updateIO(0);
    }
    timer_computation_fused.stop();

    timer_total.stop();
    double speedup = timer_computation_baseline.time() / timer_computation_fused.time();
    if(generate_report)
    {
        // Create a report
        Neon::Report report("MapFusion_" + std::string(GRID_TYPE) + "_" + std::to_string(CARDINALITY) + "D_" + std::to_string(DEVICES.size()) + "GPUs");

        //report.setToken("Token404");

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
std::vector<T> log_space(T max, T min=1, T base=2)
{
    std::vector<T> result;
    for(T val = min; val <= max; val *= base)
    {
        result.push_back(val);
    }
    return result;
}


template <typename T>
std::vector<T> lin_space(T max, T min=1, T base=1)
{
    std::vector<T> result;
    for(T val = min; val <= max; val += base)
    {
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
            int max_num_blocks = NUM_BLOCKS;
            auto all_devices = DEVICES;


            std::vector<int> fusion_factors = log_space<int>(FUSION_FACTOR);
            NUM_BLOCKS = max_num_blocks / FUSION_FACTOR;
            for(int domain_size : domain_sizes)
            {
                DOMAIN_SIZE = domain_size;
                DEVICES.clear();
                std::vector<double> speedups_gpu_count, baseline_times, fused_times;
                for(int gpu_count : gpu_counts)
                {
                    DEVICES.push_back(all_devices[gpu_count - 1]);
                    std::cout << "Starting GPU Count " << gpu_count << std::endl;
                    NUM_BLOCKS = max_num_blocks / FUSION_FACTOR;
                    std::vector<double> experiment_results;
                    if (DATA_TYPE == "single") {
                        experiment_results = mapFusionPerfTest<float>(false);
                    } else if (DATA_TYPE == "double") {
                        experiment_results = mapFusionPerfTest<double>(false);
                    } else {
                        return -1;
                    }
                    speedups_gpu_count.push_back(experiment_results[SPEEDUP]);
                    baseline_times.push_back(experiment_results[BASELINE]);
                    fused_times.push_back(experiment_results[FUSED]);
                }
                report.addMember("GPUCount_" + std::to_string(DOMAIN_SIZE), gpu_counts);
                report.addMember("SpeedupsGPUCount_" + std::to_string(DOMAIN_SIZE), speedups_gpu_count);
                report.addMember("FusedTime_" + std::to_string(DOMAIN_SIZE), fused_times);
                report.addMember("BaselineTimes_" + std::to_string(DOMAIN_SIZE), baseline_times);
            }

            DEVICES.clear();
            DEVICES.push_back(all_devices[0]);

            for(int domain_size : domain_sizes)
            {
                DOMAIN_SIZE = domain_size;
                std::vector<double> speedups_fusion_factor, baseline_times, fused_times;
                std::vector<int> fusion_factors = log_space<int>(FUSION_FACTOR);
                for(int fusion_factor : fusion_factors)
                {
                    std::cout << "Starting Fusion Factor " << fusion_factor << std::endl;
                    FUSION_FACTOR = fusion_factor;
                    NUM_BLOCKS = max_num_blocks / FUSION_FACTOR;
                    std::vector<double> experiment_results;
                    if (DATA_TYPE == "single") {
                        experiment_results = mapFusionPerfTest<float>(false);
                    } else if (DATA_TYPE == "double") {
                        experiment_results = mapFusionPerfTest<double>(false);
                    } else {
                        return -1;
                    }
                    speedups_fusion_factor.push_back(experiment_results[SPEEDUP]);
                    baseline_times.push_back(experiment_results[BASELINE]);
                    fused_times.push_back(experiment_results[FUSED]);
                }
                // Create a report

                report.addMember("FusionFactors_" + std::to_string(DOMAIN_SIZE), fusion_factors);
                report.addMember("SpeedupsFusionFactor_" + std::to_string(DOMAIN_SIZE), speedups_fusion_factor);
                report.addMember("FusedTime_" + std::to_string(DOMAIN_SIZE), fused_times);
                report.addMember("BaselineTimes_" + std::to_string(DOMAIN_SIZE), baseline_times);

                std::stringstream stringstream;
                stringstream << "Saving report file here: " << REPORT_FILENAME << std::endl;
                NEON_INFO(stringstream.str());
            }
            report.write(REPORT_FILENAME);
        }
        else
        {
            if (DATA_TYPE == "single") {
                mapFusionPerfTest<float>();
            } else if (DATA_TYPE == "double") {
                mapFusionPerfTest<double>();
            } else {
                return -1;
            }
        }
    }
    return 0;
}
