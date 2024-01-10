
#include "pressiodemoapps/euler2d.hpp"
#include "pda-schwarz/schwarz.hpp"
#include "../observer.hpp"

int main()
{
    namespace pda  = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;

    // +++++ USER INPUTS +++++
    std::string meshRoot = "./mesh";
    std::string obsRoot = "riemann2d_solution";
    const int obsFreq = 2;

    // problem definition
    const auto probId = pda::Euler2d::Riemann;
#ifdef USE_WENO5
    const auto order   = pda::InviscidFluxReconstruction::Weno5;
#elif defined USE_WENO3
    const auto order   = pda::InviscidFluxReconstruction::Weno3;
#else
    const auto order   = pda::InviscidFluxReconstruction::FirstOrder;
#endif
    const auto scheme = pode::StepScheme::CrankNicolson;
    const int icFlag = 2;
    using app_t = pdas::euler2d_app_type;

    // time stepping
    const double tf = 0.5;
    std::vector<double> dt(1, 0.02);
    const int convergeStepMax = 10;
    const double abs_err_tol = 1e-11;
    const double rel_err_tol = 1e-11;

    // +++++ END USER INPUTS +++++

    BS::thread_pool pool(8);

    // tiling, meshes, and decomposition
    auto tiling = std::make_shared<pdas::Tiling>(meshRoot);
    auto [meshObjs, meshPaths] = pdas::create_meshes(meshRoot, tiling->count());
    auto subdomains = pdas::create_subdomains<app_t>(
        meshObjs, *tiling,
		probId, scheme, order, icFlag);
    pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

    // observer
    using state_t = decltype(decomp)::state_t;
    using obs_t = FomObserver<state_t>;
    std::vector<obs_t> obsVec((*decomp.m_tiling).count());
    for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
        obsVec[domIdx] = obs_t(obsRoot + "_" + std::to_string(domIdx) + ".bin", obsFreq);
        obsVec[domIdx](::pressio::ode::StepCount(0), 0.0, *decomp.m_subdomainVec[domIdx]->getStateFull());
    }

    RuntimeObserver obs_time("runtime.bin", (*tiling).count());

    // solve
    const int numSteps = tf / decomp.m_dtMax;
    double time = 0.0;
    for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
    {
        std::cout << "Step " << outerStep << std::endl;

        // compute contoller step until convergence
        auto runtimeIter = decomp.calc_controller_step(
            outerStep,
            time,
            rel_err_tol,
            abs_err_tol,
            convergeStepMax,
            false,
	    pool
        );

        time += decomp.m_dtMax;

        // output state observer
        if ((outerStep % obsFreq) == 0) {
            const auto stepWrap = pode::StepCount(outerStep);
            for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
                obsVec[domIdx](stepWrap, time, *decomp.m_subdomainVec[domIdx]->getStateFull());
            }
        }

        // runtime observer
        obs_time(runtimeIter);

    }

    return 0;
}
