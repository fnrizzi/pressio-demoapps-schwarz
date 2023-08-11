
#include "pressiodemoapps/euler2d.hpp"
#include "pda-schwarz/schwarz.hpp"
#include "../observer.hpp"

using namespace std;

int main()
{
    namespace pda  = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;

    // +++++ USER INPUTS +++++
    string meshRoot = "./mesh";
    // string meshRoot = "/home/crwentl/research/runs/pressio/riemann/meshes/mesh_1x1";
    string obsRoot = "riemann2d_solution";
    const int obsFreq = 1;

    // problem definition
    const auto probId = pda::Euler2d::Riemann;
    const auto order  = pda::InviscidFluxReconstruction::FirstOrder;
    const auto scheme = pode::StepScheme::CrankNicolson;
    using app_t = pdas::euler_app_type;

    // time stepping
    // const double tf = 1.0;
    const double tf = 0.005;
    vector<double> dt(1, 0.005);
    const int convergeStepMax = 10;
    const double abs_err_tol = 1e-11;
    const double rel_err_tol = 1e-11;

    // +++++ END USER INPUTS +++++

    // 1. loading tiling info
    auto tiling = std::make_shared<pdas::Tiling>(meshRoot);
    tiling->describe();

    // 2. create meshes for each tile
    //    meshes must have a lifetime *longer* than that of decomp
    auto [meshPaths, meshObjs] = pdas::create_meshes(meshRoot, tiling->count());

    // 3. create "subdomain" instances
    //    note that subdomains must have a lifetime *longer* than that of decomp
    auto subdomains = pdas::create_subdomains<app_t>(meshPaths, meshObjs, *tiling, probId,
                            scheme, order, 2 /*icFlag*/);

    // 4. create decomp operating on a vector of subdomains and use it
    pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

    // observer
    // using state_t = decltype(decomp)::state_t;
    // using obs_t = FomObserver<state_t>;
    // vector<obs_t> obsVec(decomp.ndomains);
    // for (int domIdx = 0; domIdx < decomp.ndomains; ++domIdx) {
    //     obsVec[domIdx] = obs_t(obsRoot + "_" + to_string(domIdx) + ".bin", obsFreq);
    //     obsVec[domIdx](::pressio::ode::StepCount(0), 0.0, decomp.stateVec[domIdx]);
    // }

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
            convergeStepMax
        );

        time += decomp.m_dtMax;

        // output observer
        // const auto stepWrap = pode::StepCount(outerStep);
        // for (int domIdx = 0; domIdx < decomp.ndomains; ++domIdx) {
        //     obsVec[domIdx](stepWrap, time, decomp.stateVec[domIdx]);
        // }

    }

    cerr << "Finished" << endl;

    return 0;
}
