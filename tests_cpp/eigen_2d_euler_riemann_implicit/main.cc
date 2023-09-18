
#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "pressiodemoapps/euler2d.hpp"
#include "../observer.hpp"
#include <chrono>

int main()
{
    pressio::log::initialize(pressio::logto::terminal);
    pressio::log::setVerbosity({pressio::log::level::debug});

    namespace pda = pressiodemoapps;
    const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(".");
    constexpr auto order = pda::InviscidFluxReconstruction::FirstOrder;
    const auto probId = pda::Euler2d::Riemann;
    const int icFlag = 2;

    auto appObj = pda::create_problem_eigen(meshObj, probId, order, icFlag);
    using app_t = decltype(appObj);
    using state_t = typename app_t::state_type;
    using jacob_t = typename app_t::jacobian_type;

    state_t state = appObj.initialCondition();

    auto stepperObj = pressio::ode::create_implicit_stepper(
        pressio::ode::StepScheme::CrankNicolson, appObj);

    using lin_solver_t = pressio::linearsolvers::Solver<
        pressio::linearsolvers::iterative::Bicgstab, jacob_t>;
    lin_solver_t linSolverObj;
    auto NonLinSolver = pressio::create_newton_solver(stepperObj, linSolverObj);
    NonLinSolver.setStopTolerance(1e-5);

    FomObserver<state_t> Obs("riemann2d_solution.bin", 2);
    RuntimeObserver Obs_run("runtime.bin");

    const double tf = 1.0;
    const double dt = 0.005;
    const auto Nsteps = pressio::ode::StepCount(tf/dt);

    auto runtimeStart = std::chrono::high_resolution_clock::now();
    pressio::ode::advance_n_steps(stepperObj, state, 0., dt, Nsteps, Obs, NonLinSolver);
    auto runtimeEnd = std::chrono::high_resolution_clock::now();
    auto nsElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(runtimeEnd - runtimeStart).count();
    double secElapsed = static_cast<double>(nsElapsed) * 1e-9;
    Obs_run(secElapsed);

    pressio::log::finalize();
    return 0;
}
