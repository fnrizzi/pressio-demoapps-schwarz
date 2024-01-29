
#include "pressiodemoapps/swe2d.hpp"
#include "pda-schwarz/schwarz.hpp"
#include "../../observer.hpp"


int main()
{

    namespace pda  = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;

    // +++++ USER INPUTS +++++
    std::string meshRootFull = "./full_mesh_decomp";
    std::string meshRootHyper = "./sample_mesh_decomp";
    std::string obsRoot = "swe_slipWall2d_solution";
    const int obsFreq = 1;

    // problem definition
    const auto probId = pda::Swe2d::CustomBCs;
#ifdef USE_WENO5
    const auto order   = pda::InviscidFluxReconstruction::Weno5;
#elif defined USE_WENO3
    const auto order   = pda::InviscidFluxReconstruction::Weno3;
#else
    const auto order   = pda::InviscidFluxReconstruction::FirstOrder;
#endif
    const auto scheme = pode::StepScheme::BDF1;
    const int icFlag  = 1;
    using app_t = pdas::swe2d_app_type;

    // ROM definition
    std::vector<std::string> domFlagVec{"FOM", "LSPGHyper", "LSPGHyper", "FOM"};
    std::string transRoot = "./trial_space/center";
    std::string basisRoot = "./trial_space/basis";
    std::vector<int> nmodesVec(4, 25);

    // time stepping
    const double tf = 1.0;
    std::vector<double> dt(1, 0.02);
    const int convergeStepMax = 10;
    const double abs_err_tol = 1e-11;
    const double rel_err_tol = 1e-11;

    // +++++ END USER INPUTS +++++

    // tiling, meshes, and decomposition
    auto tiling = std::make_shared<pdas::Tiling>(meshRootFull);
    auto [meshObjsFull, meshPathsFull] = pdas::create_meshes(meshRootFull, tiling->count());
    std::vector<std::string> meshPathsHyper;
    for (int domIdx = 0; domIdx < meshPathsFull.size(); ++ domIdx) {
        meshPathsHyper.emplace_back(meshRootHyper + "/domain_" + std::to_string(domIdx));
    }
    auto subdomains = pdas::create_subdomains<app_t>(
        meshObjsFull, *tiling, probId, scheme, order,
        domFlagVec, transRoot, basisRoot, nmodesVec, icFlag,
        meshPathsHyper);
    pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

    // observer
    using state_t = decltype(decomp)::state_t;
    using obs_t = FomObserver<state_t>;
    std::vector<obs_t> obsVec((*decomp.m_tiling).count());
    for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
        obsVec[domIdx] = obs_t(obsRoot + "_" + std::to_string(domIdx) + ".bin", obsFreq);
        obsVec[domIdx](::pressio::ode::StepCount(0), 0.0, *decomp.m_subdomainVec[domIdx]->getStateFull());
    }

    RuntimeObserver obs_time("runtime.bin");

    // solve
    const int numSteps = tf / decomp.m_dtMax;
    double time = 0.0;
    for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
    {
        std::cout << "Step " << outerStep << std::endl;

        // compute contoller step until convergence
        auto runtimeStart = std::chrono::high_resolution_clock::now();
        auto numSubiters = decomp.calc_controller_step(
            pdas::SchwarzMode::Multiplicative,
            outerStep,
            time,
            rel_err_tol,
            abs_err_tol,
            convergeStepMax
        );
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

        // runtime observer
        obs_time(secsElapsed, numSubiters);

    }

    return 0;

}
