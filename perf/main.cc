
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

auto get_flux_enum(int stencilsize){
  pda::InviscidFluxReconstruction fluxE;
  if (stencilsize == 3){
    fluxE = pda::InviscidFluxReconstruction::FirstOrder;
  }
  else if (stencilsize == 5){
    fluxE = pda::InviscidFluxReconstruction::Weno3;
  }
  else{
    throw std::runtime_error("invalid order");
  }
  return fluxE;
}

#if defined SCHWARZ_PERF_A2
template<class app_t>
void run_a2(const std::string & ml,
	   BS::thread_pool & pool,
	   pda::InviscidFluxReconstruction invFluxRec,
	   int numSteps)
{
  std::string meshRoot = "./mesh_" + ml + "/mesh";
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

#if defined(SCHWARZ_PERF_B) || defined(SCHWARZ_PERF_C) \
  || defined(SCHWARZ_PERF_D) || defined(SCHWARZ_PERF_E) || defined(SCHWARZ_PERF_F)
template<class app_t>
void realrun(const std::string & ml,
	     BS::thread_pool & pool,
	     pda::InviscidFluxReconstruction invFluxRec,
	     int numSteps,
	     int convStepMax)
{
  std::string meshRoot = "./mesh_" + ml + "/mesh";
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

#if defined(SCHWARZ_PERF_A1)

void run_a1(const std::string & mesh_path,
	    int numsubs, int ssize, int numsteps)
{
  const auto probId  = pda::Swe2d::SlipWall;
  const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(mesh_path);

  using app_t = decltype(pda::create_problem_eigen(meshObj, probId, get_flux_enum(ssize), 1));
  using v_t = typename app_t::state_type;
  using m_t = typename app_t::jacobian_type;

  int numth = 0;
  double avgruntime = 0.;
  std::vector<std::shared_ptr<app_t>> apps(numsubs, nullptr);
#pragma omp parallel reduction (+:avgruntime)
  {
    #pragma omp single
    {
      numth = omp_get_num_threads();
    }

    v_t s;
    v_t f;
    m_t j;

    #pragma omp for schedule(static)
    for (int i=0; i<numsubs; ++i){
      apps[i] = std::make_shared<app_t>(pda::create_problem_eigen(meshObj, probId, get_flux_enum(ssize), 1));
      s = apps[i]->initialCondition();
      f = apps[i]->createRhs();
      j = apps[i]->createJacobian();
    }

    auto runtimeStart = std::chrono::high_resolution_clock::now();
    auto lam = [&](int i){
      for (int k=0; k<numsteps; ++k){
	(*apps[i])(s, 0., f, j, true);
      }
    };

    #pragma omp for schedule(static)
    for (int i=0; i<numsubs; ++i){
      lam(i);
    }
    const auto runtimeEnd = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(runtimeEnd - runtimeStart);
    const double elapsed = static_cast<double>(duration.count());
    avgruntime += elapsed/numsteps;
  }//parallel

  std::cout << "time: ms/threads = " << avgruntime/numth << ' '<< numth << '\n';
}
#endif

int read_stencil_size(const std::string & mesh_path)
{
  const std::string filename = mesh_path + "/info.dat";
  std::ifstream foundFile(filename);
  if(!foundFile){
    std::cout << "file not found " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  int ssize = 0;
  std::ifstream source( filename, std::ios_base::in);
  std::string line;
  while (std::getline(source, line) )
    {
      std::istringstream ss(line);
      std::string colVal;
      ss >> colVal;

      if (colVal == "stencilSize"){
	ss >> colVal;
	ssize = std::stoi(colVal);
      }
    }
  source.close();
  return ssize;
}

// =============================================
int main(int argc, char *argv[])
// =============================================
{

#if defined(SCHWARZ_PERF_A1)
  const std::string mesh_path = argv[1];
  const int numsubs  = std::stoi(argv[2]);
  const int numSteps = std::stoi(argv[3]);
  const int ssize = read_stencil_size(mesh_path);

  std::cout << "mesh_path = " << mesh_path << ' '
	    << ", numsubs = " << numsubs << ' '
	    << ", ssize = " << ssize << ' '
	    << ", numSteps = " << numSteps << '\n';
  run_a1(mesh_path, numsubs, ssize, numSteps);
#endif

// #if defined(SCHWARZ_PERF_A2)
//   const int nt = parse_num_threads(argc, argv);
//   const std::string mesh_label = argv[2];
//   const int order = std::stoi(argv[3]);
//   const int numSteps = std::stoi(argv[4]);
//   BS::thread_pool pool(nt);

//   std::cout << "nthreads = " << nt << ' '
// 	    << ", order = " << order << ' '
// 	    << ", numSteps = " << numSteps << '\n';
//   run_a2<pdas::swe2d_app_type>(mesh_label, pool, get_flux_enum(order), numSteps);
// #endif

// #if defined(SCHWARZ_PERF_B) || defined(SCHWARZ_PERF_C) \
//   || defined(SCHWARZ_PERF_D) || defined(SCHWARZ_PERF_E) || defined(SCHWARZ_PERF_F)

//   const int nt = parse_num_threads(argc, argv);
//   const std::string mesh_label = argv[2];
//   const int order = std::stoi(argv[3]);
//   const int numSteps = std::stoi(argv[4]);
//   const int convStepMax = std::stoi(argv[5]);
//   std::cout << "nthreads = " << nt << ' '
// 	    << ", order = " << order << ' '
// 	    << ", numSteps = " << numSteps << ' '
// 	    << ", convStepMax = " << convStepMax << '\n';

//   BS::thread_pool pool(nt);
//   realrun<pdas::swe2d_app_type>(mesh_label, pool, get_flux_enum(order),
// 				numSteps, convStepMax);
// #endif

  return 0;
}
