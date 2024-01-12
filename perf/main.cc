
#include "pressiodemoapps/swe2d.hpp"
#include "pda-schwarz/schwarz.hpp"

namespace pda  = pressiodemoapps;
namespace pdas = pdaschwarz;
namespace pode = pressio::ode;

int parse_num_threads(int argc, char *argv[])
{
  /*
    1. no args passed => 1 thread is used
    2. ./exe n        => n threads are used if n!=1
    3. ./exe -1       => max hardware concurrency
  */

  int numthreads = 1;
  if (argc >= 2){
    numthreads = std::stoi(argv[1]);
    if (numthreads == -1){ numthreads = std::thread::hardware_concurrency(); }
    assert(numthreads >= 1);
  }
  else {
    std::cout << "defaulting to 1 thread because not cmdline arg was found\n";
  }

  return numthreads;
}

#if defined SCHWARZ_PERF_A
template<class app_t>
void run_A(BS::thread_pool & pool,
	   pda::InviscidFluxReconstruction invFluxRec,
	   int numSteps)
{
  std::string meshRoot = "./mesh_a/mesh";
  const auto probId = pda::Swe2d::CustomBCs;

  const auto scheme = pode::StepScheme::BDF1;
  const int icFlag = 1;

  /*dt is not actually unused below so does not matter, just use something*/
  std::vector<double> dt(1, 0.1);

  auto tiling = std::make_shared<pdas::Tiling>(meshRoot);
  auto [meshObjs, meshPaths] = pdas::create_meshes(meshRoot, tiling->count());
  auto subdomains = pdas::create_subdomains<
    app_t>(meshObjs, *tiling, probId, scheme, invFluxRec, icFlag);
  pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

  // warmup first
  decomp._testonlydemoappsevaluation(pool);

  // timed loop
  std::cout << "numSteps " << numSteps << std::endl;
  auto runtimeStart = std::chrono::high_resolution_clock::now();
  for (int outerStep = 1; outerStep <= numSteps; ++outerStep){
    decomp._testonlydemoappsevaluation(pool);
  }
  const auto runtimeEnd = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(runtimeEnd - runtimeStart);
  double elapsed = static_cast<double>(duration.count());
  std::cout << "proxyrun: ms/step = " << elapsed/numSteps << '\n';
}
#endif

#if defined(SCHWARZ_PERF_B) || defined(SCHWARZ_PERF_C) || defined(SCHWARZ_PERF_D) || defined(SCHWARZ_PERF_E) || defined(SCHWARZ_PERF_F)
template<class app_t>
void realrun(BS::thread_pool & pool,
	     pda::InviscidFluxReconstruction invFluxRec,
	     int numSteps,
	     int convStepMax)
{
  std::string meshRoot = "./mesh_a/mesh";
  const auto probId = pda::Swe2d::CustomBCs;

  const auto scheme = pode::StepScheme::BDF1;
  const int icFlag = 1;

  std::vector<double> dt(1, 0.02);
  const double abs_err_tol = 1e-11;
  const double rel_err_tol = 1e-11;

  auto tiling = std::make_shared<pdas::Tiling>(meshRoot);
  auto [meshObjs, meshPaths] = pdas::create_meshes(meshRoot, tiling->count());
  auto subdomains = pdas::create_subdomains<
    app_t>(meshObjs, *tiling, probId, scheme, invFluxRec, icFlag);
  pdas::SchwarzDecomp decomp(subdomains, tiling, dt);

  // do warmup first
  double time = 0.0;
  for (int outerStep = 1; outerStep <= 2; ++outerStep){
    auto runtimeIter = decomp.calc_controller_step(
			   pdas::SchwarzMode::Additive,
			   outerStep, time, rel_err_tol, abs_err_tol,
			   convStepMax, pool);
    time += decomp.m_dtMax;
  }

  // timed loop
  std::cout << "numSteps " << numSteps << std::endl;
  time = 0.0;
  auto runtimeStart = std::chrono::high_resolution_clock::now();
  for (int outerStep = 1; outerStep <= numSteps; ++outerStep)
  {
    auto runtimeIter = decomp.calc_controller_step(
			   pdas::SchwarzMode::Additive,
			   outerStep, time,
			   rel_err_tol, abs_err_tol,
			   convStepMax, pool);

    time += decomp.m_dtMax;
  }

  const auto runtimeEnd = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(runtimeEnd - runtimeStart);
  double elapsed = static_cast<double>(duration.count());
  std::cout << "realrun: ms/step = " << elapsed/numSteps << '\n';
}
#endif



int main(int argc, char *argv[])
{
  const int nt = parse_num_threads(argc, argv);
  const int order = std::stoi(argv[2]);
  const int numSteps = std::stoi(argv[3]);
  std::cout << "nthreads = " << nt << '\n';
  std::cout << "order = " << order << '\n';
  std::cout << "numSteps = " << numSteps << '\n';
  BS::thread_pool pool(nt);


  pda::InviscidFluxReconstruction fluxE;
  if (order == 1){
    fluxE = pda::InviscidFluxReconstruction::FirstOrder;
  }
  else if (order==3){
    fluxE = pda::InviscidFluxReconstruction::Weno3;
  }
  else{
    throw std::runtime_error("invalid order");
  }

#if defined(SCHWARZ_PERF_A)
  run_A<pdas::swe2d_app_type>(pool, fluxE, numSteps);
#endif

#if defined(SCHWARZ_PERF_B) || defined(SCHWARZ_PERF_C) || defined(SCHWARZ_PERF_D) || defined(SCHWARZ_PERF_E) || defined(SCHWARZ_PERF_F)
  const int convStepMax = std::stoi(argv[4]);
  std::cout << "convStepMax = " << convStepMax << '\n';
  realrun<pdas::swe2d_app_type>(pool, fluxE, numSteps, convStepMax);
#endif

  return 0;
}
