#include <chrono>
#include "pressiodemoapps/swe2d.hpp"
#include "pda-schwarz/schwarz.hpp"
#include "../../observer.hpp"
#include "../../help_cmdline.hpp"

int main(int argc, char *argv[])
{
    namespace pda  = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;

#if defined SCHWARZ_ENABLE_THREADPOOL
    const int numthreads = parse_num_threads(argc, argv);
    std::string dir_suffix = "_tp";
#elif defined SCHWARZ_ENABLE_OMP
    std::string dir_suffix = "_omp";
# else
    std::string dir_suffix = "";
#endif

    // +++++ USER INPUTS +++++

    const int obsFreq = 1;

    const auto probId = pda::Swe2d::CustomBCs;
#ifdef USE_WENO5
    static_assert(false);
    const auto order   = pda::InviscidFluxReconstruction::Weno5;
#elif defined USE_WENO3
    std::string outRoot = "./weno3" + dir_suffix;
    const auto order   = pda::InviscidFluxReconstruction::Weno3;
#else
    std::string outRoot = "./firstorder" + dir_suffix;
    const auto order   = pda::InviscidFluxReconstruction::FirstOrder;
#endif
    std::string obsRoot = outRoot + "/swe_slipWall2d_solution";
    std::string meshRoot = outRoot + "/mesh";

    const auto scheme = pode::StepScheme::BDF1;
    const int icFlag = 1;
    using app_t = pdas::swe2d_app_type;

    const double tf = 1.0;
    std::vector<double> dt(1, 0.02);
    const int convergeStepMax = 10;
    const double abs_err_tol = 1e-11;
    const double rel_err_tol = 1e-11;

    // +++++ END USER INPUTS +++++

    auto tiling = std::make_shared<pdas::Tiling>(meshRoot);
    auto [meshes, meshPaths] = pdas::create_meshes(meshRoot, tiling->count());
    auto subdomains = pdas::create_subdomains<app_t>(
        meshes, *tiling, probId, scheme, order, icFlag);
    pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

    // observers
    using state_t = decltype(decomp)::state_t;
    using obs_t = FomObserver<state_t>;
    std::vector<obs_t> obsVec((*decomp.m_tiling).count());
    for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
        obsVec[domIdx] = obs_t(obsRoot + "_" + std::to_string(domIdx) + ".bin", obsFreq);
        obsVec[domIdx](::pressio::ode::StepCount(0), 0.0, *decomp.m_subdomainVec[domIdx]->getStateFull());
    }
    RuntimeObserver obs_time(outRoot + "/runtime.bin");

// -----------------------------------------
// OMP
// -----------------------------------------
#if defined SCHWARZ_ENABLE_OMP

    const int numSteps = tf / decomp.m_dtMax;
    std::chrono::time_point<std::chrono::high_resolution_clock> runtimeStart;
    double secsElapsed;

#pragma omp parallel firstprivate(numSteps, rel_err_tol, abs_err_tol, convergeStepMax)
{

    // auto loop = [&](int outerStep, double simultime){
    //     decomp.additive_step(outerStep, simultime, rel_err_tol, abs_err_tol, convergeStepMax);
    // };

    double simultime = 0.0;
    for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
    {
#pragma omp single
        {
            std::cout << "Step " << outerStep << std::endl;
        }

        // execute Schwarz iteration
        // loop(outerStep, simultime);

#pragma omp barrier
#pragma omp master
        {
            auto runtimeStart = std::chrono::high_resolution_clock::now();
        }
        auto numSubiters = decomp.additive_step(outerStep, simultime, rel_err_tol, abs_err_tol, convergeStepMax);
#pragma omp barrier
#pragma omp master
        {
            const auto runtimeEnd = std::chrono::high_resolution_clock::now();
            const auto nsDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(runtimeEnd - runtimeStart);
            secsElapsed = static_cast<double>(nsDuration.count());
        }

#pragma omp barrier
#pragma omp master
        {
            // output observer
            if ((outerStep % obsFreq) == 0) {
                const auto stepWrap = pode::StepCount(outerStep);
                for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
                    obsVec[domIdx](stepWrap, time, *decomp.m_subdomainVec[domIdx]->getStateFull());
                }
            }
            obs_time(secsElapsed, numSubiters);
        }

    }
}

// -----------------------------------------
// Thread pool
// -----------------------------------------
#elif defined SCHWARZ_ENABLE_THREADPOOL

    // solve
    BS::thread_pool pool(numthreads);
    const int numSteps = tf / decomp.m_dtMax;
    double time = 0.0;
    for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
    {
        std::cout << "Step " << outerStep << std::endl;

        // compute contoller step until convergence
        auto runtimeStart = std::chrono::high_resolution_clock::now();
        auto numSubiters = decomp.additive_step(outerStep, time, rel_err_tol,
                             abs_err_tol, convergeStepMax, pool);
        const auto runtimeEnd = std::chrono::high_resolution_clock::now();
        const auto nsDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(runtimeEnd - runtimeStart);
        const double secsElapsed = static_cast<double>(nsDuration.count()) * 1e-9;

        time += decomp.m_dtMax;

        // output observer
        if ((outerStep % obsFreq) == 0) {
            const auto stepWrap = pode::StepCount(outerStep);
            for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
                obsVec[domIdx](stepWrap, time, *decomp.m_subdomainVec[domIdx]->getStateFull());
            }
        }
        obs_time(secsElapsed, numSubiters);
    }

#endif

    return 0;
}
