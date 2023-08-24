
#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "pressiodemoapps/swe2d.hpp"
#include "../../observer.hpp"
#include "pressio/rom_subspaces.hpp"
#include "pressio/rom_galerkin_unsteady.hpp"
#include "pda-schwarz/rom_utils.hpp"

using namespace std;

int main()
{
    pressio::log::initialize(pressio::logto::terminal);
    pressio::log::setVerbosity({pressio::log::level::debug});

    namespace pda = pressiodemoapps;
    namespace pdas = pdaschwarz;
    namespace pode = pressio::ode;
    namespace prom = pressio::rom;
    namespace pgal = pressio::rom::galerkin;

    const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(".");

    string transfile = "../../trial_space/center.bin";
    string basisfile = "../../trial_space/basis.bin";
    const int nmodes = 25;

    constexpr auto order = pda::InviscidFluxReconstruction::FirstOrder;
    auto scheme = pode::StepScheme::BDF1;

    // create fomSystem
    const auto probId  = pda::Swe2d::SlipWall;
    auto fomSystem = pda::create_problem_eigen(meshObj, probId, order);

    using FomSystemType = decltype(fomSystem);
    using scalar_type = FomSystemType::scalar_type;
    using reduced_state_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;

    // read and define trial space
    auto trans = pdas::read_vector_from_binary<scalar_type>(transfile);
    auto basis = pdas::read_matrix_from_binary<scalar_type>(basisfile, nmodes);
    const auto trialSpace = prom::create_trial_column_subspace<
        reduced_state_type>(move(basis), move(trans), true);
    auto reducedState = trialSpace.createReducedState();

    // define ROM problem
    auto problem = pgal::create_unsteady_implicit_problem(scheme, trialSpace, fomSystem);

    // define solver
    using hessian_t       = Eigen::Matrix<scalar_type, -1, -1>;
    using solver_tag      = pressio::linearsolvers::direct::HouseholderQR;
    using linear_solver_t = pressio::linearsolvers::Solver<solver_tag, hessian_t>;
    linear_solver_t linearSolver;

    auto solver = pressio::create_newton_solver(problem, linearSolver);
    solver.setStopTolerance(1e-5);

    // observer
    using app_t = decltype(fomSystem);
    StateObserver Obs("swe_slipWall2d_solution.bin", 10);

    const double tf = 10.0;
    const double dt = 0.01;
    const auto Nsteps = pressio::ode::StepCount(tf/dt);
    pode::advance_n_steps(problem, reducedState, 0.0, dt, Nsteps, Obs, solver);

    pressio::log::finalize();
    return 0;
}
