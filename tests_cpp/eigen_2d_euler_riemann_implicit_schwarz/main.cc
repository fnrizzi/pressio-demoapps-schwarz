
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

    // time stepping
    // const double tf = 1.0;
    const double tf = 0.005;
    vector<double> dt(1, 0.005);
    const int convergeStepMax = 10;
    const double abs_err_tol = 1e-11;
    const double rel_err_tol = 1e-11;

    // +++++ END USER INPUTS +++++

    using mesh_t = pda::cellcentered_uniform_mesh_eigen_type;
    using app_t = decltype(
        pda::create_problem_eigen(declval<mesh_t>(), probId, order,
        pdas::BCFunctor(pdas::BCType::Dummy), pdas::BCFunctor(pdas::BCType::Dummy), pdas::BCFunctor(pdas::BCType::Dummy), pdas::BCFunctor(pdas::BCType::Dummy), 1)
    );
    using subdom_t = typename pdas::Subdomain<decltype(probId), decltype(order), decltype(scheme), mesh_t, app_t>;

    // decomposition
    auto decomp = pdas::SchwarzDecomp<
        decltype(probId),
        mesh_t,
        decltype(order),
        decltype(scheme),
        subdom_t
    >(probId, order, scheme, meshRoot, dt, 2);

    // observer
    // using state_t = decltype(decomp)::state_t;
    // using obs_t = FomObserver<state_t>;
    // vector<obs_t> obsVec(decomp.ndomains);
    // for (int domIdx = 0; domIdx < decomp.ndomains; ++domIdx) {
    //     obsVec[domIdx] = obs_t(obsRoot + "_" + to_string(domIdx) + ".bin", obsFreq);
    //     obsVec[domIdx](::pressio::ode::StepCount(0), 0.0, decomp.stateVec[domIdx]);
    // }

    // solve
    const int numSteps = tf / decomp.dtMax;
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

        time += decomp.dtMax;

        // output observer
        // const auto stepWrap = pode::StepCount(outerStep);
        // for (int domIdx = 0; domIdx < decomp.ndomains; ++domIdx) {
        //     obsVec[domIdx](stepWrap, time, decomp.stateVec[domIdx]);
        // }

    }

    cerr << "Finished" << endl;

    return 0;
}
