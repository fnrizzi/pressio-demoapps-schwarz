
#include "pressiodemoapps/swe2d.hpp"
#include "pda-schwarz/schwarz.hpp"
#include "../../observer.hpp"
// #include "pressio/rom_subspaces.hpp"
// #include "pressio/rom_lspg_unsteady.hpp"
// #include "pda-schwarz/rom_utils.hpp"

using namespace std;

int main()
{

    namespace pda  = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;

    // +++++ USER INPUTS +++++
    string meshRoot = "./mesh";
    string obsRoot = "swe_slipWall2d_solution";
    const int obsFreq = 10;

    // problem definition
    const auto probId = pda::Swe2d::CustomBCs;
    const auto order  = pda::InviscidFluxReconstruction::FirstOrder;
    const auto scheme = pode::StepScheme::BDF1;
    const int icFlag  = 1;
    using app_t = pdas::swe2d_app_type;

    // ROM definition
    vector<string> domFlagVec(4, "LSPG");
    string transRoot = "./trial_space/center";
    string basisRoot = "./trial_space/basis";
    vector<int> nmodesVec(4, 25);

    // time stepping
    const double tf = 5.0;
    vector<double> dt(1, 0.01);
    const int convergeStepMax = 10;
    const double abs_err_tol = 1e-11;
    const double rel_err_tol = 1e-11;

    // +++++ END USER INPUTS +++++

    // tiling, meshes, and decomposition
    auto tiling = std::make_shared<pdas::Tiling>(meshRoot);
    auto [meshPaths, meshObjs] = pdas::create_meshes(meshRoot, tiling->count());
    auto subdomains = pdas::create_subdomains<app_t>(
        meshPaths, meshObjs, *tiling, probId, scheme, order,
        domFlagVec, transRoot, basisRoot, nmodesVec, icFlag);
    pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

    // observer
    using state_t = decltype(decomp)::state_t;
    using obs_t = FomObserver<state_t>;
    vector<obs_t> obsVec((*decomp.m_tiling).count());
    for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
        obsVec[domIdx] = obs_t(obsRoot + "_" + to_string(domIdx) + ".bin", obsFreq);
        obsVec[domIdx](::pressio::ode::StepCount(0), 0.0, decomp.m_subdomainVec[domIdx]->m_state);
    }

    // solve
    const int numSteps = tf / decomp.m_dtMax;
    double time = 0.0;
    for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
    {
        cout << "Step " << outerStep << endl;

        // compute contoller step until convergence
        decomp.calc_controller_step(
            outerStep,
            time,
            rel_err_tol,
            abs_err_tol,
            convergeStepMax,
            false
        );

        time += decomp.m_dtMax;

        // output observer
        if ((outerStep % obsFreq) == 0) {
            const auto stepWrap = pode::StepCount(outerStep);
            for (int domIdx = 0; domIdx < (*decomp.m_tiling).count(); ++domIdx) {
                obsVec[domIdx](stepWrap, time, decomp.m_subdomainVec[domIdx]->m_state);
            }
        }

    }

    return 0;

}
