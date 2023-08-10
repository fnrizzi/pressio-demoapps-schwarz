
#include "pressio/ode_steppers_explicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "pressiodemoapps/euler2d.hpp"
#include "pda-schwarz/schwarz_frizzi.hpp"
#include "../observer.hpp"

int main()
{
  namespace pda  = pressiodemoapps;
  namespace pdas = pdaschwarz;
  namespace pode = pressio::ode;
  namespace plog = pressio::log;

  plog::initialize(pressio::logto::terminal);
  plog::setVerbosity({pressio::log::level::debug});

  // ++++++++++++++++++++++++++++++++++++++++
  // +++++ USER INPUTS +++++
  // ++++++++++++++++++++++++++++++++++++++++
  string meshRoot = "./mesh";
  string obsRoot = "riemann2d_solution";
  const int obsFreq = 1;

  // problem definition
  const auto probId = pda::Euler2d::Riemann;
  const auto order  = pda::InviscidFluxReconstruction::FirstOrder;
  const auto scheme = pode::StepScheme::CrankNicolson;
  using app_t = pdas::euler_app_type; // this is a helper alias

  // time stepping
  // const double tf = 1.0;
  const double tf = 0.005;
  std::vector<double> dt(1, 0.005);
  const int convergeStepMax = 10;
  const double abs_err_tol = 1e-11;
  const double rel_err_tol = 1e-11;
  // ++++++++++++++++++++++++++++++++++++++++
  // +++++ END USER INPUTS +++++
  // ++++++++++++++++++++++++++++++++++++++++

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
  pdas::SchwarzDecompFrizzi decomp(subdomains, tiling, dt);
  const int numSteps = tf / decomp.m_dtMax;
  double time = 0.0;
  for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
    {
      cout << "Step " << outerStep << endl;
      decomp.calc_controller_step(outerStep, time, rel_err_tol,
				  abs_err_tol, convergeStepMax);
      time += decomp.m_dtMax;
    }

  // these two scopes below are just things I tried and work
  // {
  //   // explicit only
  //   auto & subdom = decomp.m_subdomainVec[0];
  //   auto & appObj = *(subdom.m_app);
  //   auto state = appObj.initialCondition();
  //   FomObserver<decltype(state)> Obs("riemann2d_solution.bin", 10);
  //   const auto mydt = 0.001;
  //   const auto Nsteps = pressio::ode::StepCount(0.6/mydt);
  //   auto stepperObj = pressio::ode::create_ssprk3_stepper(appObj);
  //   pressio::ode::advance_n_steps(stepperObj, state, 0., mydt, Nsteps, Obs);
  // }

  //  {
  //    //   implicit
  //   auto & subdom = decomp.m_subdomainVec[0];
  //   auto & appObj = *(subdom.m_app);

  //   auto state = appObj.initialCondition();
  //   FomObserver<decltype(state)> Obs("riemann2d_solution.bin", 10);
  //   const auto mydt = 0.001;
  //   const auto Nsteps = pressio::ode::StepCount(0.6/mydt);
  //   pressio::ode::advance_n_steps(subdom.m_stepper, state, 0., mydt,
  // 				  Nsteps, Obs, subdom.m_nonlinSolver);
  // }

  plog::finalize();
  return 0;
}
