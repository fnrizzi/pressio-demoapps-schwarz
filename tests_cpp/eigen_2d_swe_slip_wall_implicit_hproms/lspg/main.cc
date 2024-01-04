
#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "pressiodemoapps/swe2d.hpp"
#include "../../observer.hpp"
#include <chrono>
#include "pressio/rom_subspaces.hpp"
#include "pressio/rom_lspg_unsteady.hpp"
#include "pda-schwarz/rom_utils.hpp"

int main()
{
    pressio::log::initialize(pressio::logto::terminal);
    pressio::log::setVerbosity({pressio::log::level::debug});

    namespace pda = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;
    namespace prom = pressio::rom;
    namespace plspg = pressio::rom::lspg;
    namespace pnlins = pressio::nonlinearsolvers;

    const auto meshObjFull = pda::load_cellcentered_uniform_mesh_eigen("./full_mesh");
    const auto meshObjHyp = pda::load_cellcentered_uniform_mesh_eigen("./sample_mesh");

    std::string transfile = "./trial_space/center.bin";
    std::string basisfile = "./trial_space/basis.bin";
    const int nmodes = 25;
    std::string stencilFile = "./sample_mesh/stencil_mesh_gids.dat";
    std::string sampleFile = "./sample_mesh/sample_mesh_gids.dat";

#ifdef USE_WENO5
    const auto order   = pda::InviscidFluxReconstruction::Weno5;
#elif defined USE_WENO3
    const auto order   = pda::InviscidFluxReconstruction::Weno3;
#else
    const auto order   = pda::InviscidFluxReconstruction::FirstOrder;
#endif
    auto scheme = pode::StepScheme::BDF1;
    const int icFlag = 1;

    // create fomSystems (full system needed for projecting initial conditions)
    const auto probId  = pda::Swe2d::SlipWall;
    auto fomSystemFull = pda::create_problem_eigen(meshObjFull, probId, order, icFlag);
    auto fomSystemHyp = pda::create_problem_eigen(meshObjHyp, probId, order, icFlag);
    const auto numDofsPerCell = fomSystemFull.numDofPerCell();

    using FomSystemType = decltype(fomSystemFull);
    using scalar_type = FomSystemType::scalar_type;
    using reduced_state_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;

    // read and define full and sampled trial spaces
    auto trans = pdas::read_vector_from_binary<scalar_type>(transfile);
    auto basis = pdas::read_matrix_from_binary<scalar_type>(basisfile, nmodes);
    const auto stencilGids = pdas::create_cell_gids_vector_and_fill_from_ascii(stencilFile);
    auto transHyp = pdas::reduce_vector_on_stencil_mesh(trans, stencilGids, numDofsPerCell);
    auto basisHyp = pdas::reduce_matrix_on_stencil_mesh(basis, stencilGids, numDofsPerCell);
    const auto trialSpaceFull = prom::create_trial_column_subspace<
       reduced_state_type>(std::move(basis), std::move(trans), true);
    const auto trialSpaceHyp = prom::create_trial_column_subspace<
       reduced_state_type>(std::move(basisHyp), std::move(transHyp), true);

    // project initial condition
    auto state = fomSystemFull.initialCondition();
    auto u = pressio::ops::clone(state);
    pressio::ops::update(u, 0., state, 1, trialSpaceFull.translationVector(), -1);
    auto reducedState = trialSpaceFull.createReducedState();
    pressio::ops::product(::pressio::transpose(), 1., trialSpaceFull.basisOfTranslatedSpace(), u, 0., reducedState);

    // hyper-reduction updater
    pdas::HypRedUpdater<scalar_type> hrUpdater(numDofsPerCell, stencilFile, sampleFile);

    // define ROM problem
    auto problem = plspg::create_unsteady_problem(scheme, trialSpaceHyp, fomSystemHyp, hrUpdater);
    auto stepper = problem.lspgStepper();

    // define solver
    using hessian_t       = Eigen::Matrix<scalar_type, -1, -1>;
    using solver_tag      = pressio::linearsolvers::direct::HouseholderQR;
    using linear_solver_t = pressio::linearsolvers::Solver<solver_tag, hessian_t>;
    linear_solver_t linearSolver;

    auto solver = pressio::create_gauss_newton_solver(stepper, linearSolver);
    solver.setStopCriterion(pnlins::Stop::WhenAbsolutel2NormOfGradientBelowTolerance);
    solver.setStopTolerance(1e-5);

    // observer
    StateObserver Obs("swe_slipWall2d_solution.bin", 1);
    RuntimeObserver Obs_run("runtime.bin");

    const double tf = 1.0;
    const double dt = 0.02;
    const auto Nsteps = pressio::ode::StepCount(tf/dt);

    auto runtimeStart = std::chrono::high_resolution_clock::now();
    pode::advance_n_steps(stepper, reducedState, 0.0, dt, Nsteps, Obs, solver);
    auto runtimeEnd = std::chrono::high_resolution_clock::now();
    auto nsElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(runtimeEnd - runtimeStart).count();
    double secElapsed = static_cast<double>(nsElapsed) * 1e-9;
    Obs_run(secElapsed);

    pressio::log::finalize();
    return 0;
}
