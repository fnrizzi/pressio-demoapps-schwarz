
#ifndef PRESSIODEMOAPPS_SCHWARZ_FRIZZI_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_FRIZZI_HPP_

#include "pressio/ode_steppers_implicit.hpp"
#include "pressiodemoapps/impl/ghost_relative_locations.hpp"
#include "./custom_bcs.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

namespace pdaschwarz{

//
// aliases to make life easier
//
using mesh_t = pressiodemoapps::cellcentered_uniform_mesh_eigen_type;
using euler_app_type =
  decltype(pressiodemoapps::create_problem_eigen(declval<mesh_t>(),
						 std::declval<pressiodemoapps::Euler2d>(),
						 std::declval<pda::InviscidFluxReconstruction>(),
						 // NOTE: bc functors must be rvalues because in the constructor
						 // of SubdomainFrizzi we construct the app by passing rvalues
						 std::declval<BCFunctor>(),
						 std::declval<BCFunctor>(),
						 std::declval<BCFunctor>(),
						 std::declval<BCFunctor>(),
						 int() /* initial condition */
						 )
	   );



//
// auxiliary functions to read *full* mesh dims from files
//
std::array<int,3> read_mesh_dims(const std::string & meshPath, int domIdx)
{
  // Read uniform mesh dimensions, as this was removed from PDA
  // NOTE: this should be made obsolete when code is generalized
  // to connectivity graph, doesn't depend on uniform mesh

  const auto inFile = meshPath + "/info.dat";
  ifstream foundFile(inFile);
  if(!foundFile){
    cout << "file not found " << inFile << endl;
    exit(EXIT_FAILURE);
  }

  std::array<int, 3> dims;
  ifstream source( inFile, ios_base::in);
  string line;
  while (getline(source, line) ){
    istringstream ss(line);
    string colVal;
    ss >> colVal;
    if      (colVal == "nx"){ ss >> colVal; dims[0] = stoi(colVal); }
    else if (colVal == "ny"){ ss >> colVal; dims[1] = stoi(colVal); }
    else if (colVal == "nz"){ ss >> colVal; dims[2] = stoi(colVal); }
  }
  source.close();
  return dims;
}


//
// function to create a vector of meshes given a count and meshRoot
//
auto create_meshes(std::string const & meshRoot, const int n)
{
  namespace pda = pressiodemoapps;
  using mesh_t = pda::cellcentered_uniform_mesh_eigen_type;
  std::vector<mesh_t> meshes;
  std::vector<std::string> meshPaths;
  for (int i=0; i<n; ++i){
    meshPaths.emplace_back(meshRoot + "/domain_" + std::to_string(i));
    meshes.emplace_back( pda::load_cellcentered_uniform_mesh_eigen(meshPaths.back()) );
  }
  return std::pair(meshPaths, meshes);
}


//
// struct to hold topology information about tiling/partitions
//
struct Tiling
{
  explicit Tiling(const std::string & meshRoot){
    read_domain_info(meshRoot);
    calc_neighbor_dims();
  }

  void describe(){
    std::cout << " Tiling info: "
	      << " ndomX = " << m_ndomX
	      << " ndomY = " << m_ndomY
	      << " ndomZ = " << m_ndomZ
	      << " ndomains = " << m_ndomains
	      << " overlap = " << m_overlap
	      << '\n';
  }

  int dim() const{ return m_dim; }
  int overlap() const { return m_overlap; }
  auto exchDomIdVec() const{ return m_exchDomIdVec; }
  int count() const { return m_ndomains; }
  int countX() const { return m_ndomX; }
  int countY() const { return m_ndomY; }
  int countZ() const { return m_ndomZ; }

private:
  void read_domain_info(const string & meshRoot)
  {
    const auto inFile = meshRoot + "/info_domain.dat";
    ifstream foundFile(inFile);
    if(!foundFile){
      cout << "file not found " << inFile << endl;
      exit(EXIT_FAILURE);
    }

    // defaults
    m_ndomX = 1;
    m_ndomY = 1;
    m_ndomZ = 1;
    ifstream source( inFile, ios_base::in);
    string line;
    while (getline(source, line) )
      {
	istringstream ss(line);
	string colVal;
	ss >> colVal;

	if (colVal == "dim"){
	  ss >> colVal; m_dim = stoi(colVal);
	  if (m_dim < 1) throw runtime_error("dim must be >= 1");
	}
	else if (colVal == "ndomX"){
	  ss >> colVal; m_ndomX = stoi(colVal);
	  if (m_ndomX < 1) throw runtime_error("ndomX must be >= 1");
	}

	else if (colVal == "ndomY"){
	  ss >> colVal; m_ndomY = stoi(colVal);
	  if (m_ndomY < 1) throw runtime_error("ndomY must be >= 1");
	}
	else if (colVal == "ndomZ"){
	  ss >> colVal; m_ndomZ = stoi(colVal);
	  if (m_ndomZ < 1) throw runtime_error("ndomZ must be >= 1");
	}

	else if (colVal == "overlap"){
	  ss >> colVal;
	  m_overlap = stoi(colVal);
	  if (m_overlap < 0) throw runtime_error("overlap must be > 0");

	  // has to be an even number for simplicity, can change later
	  if (m_overlap % 2) {
	    cerr << "overlap must be an even number" << endl;
	    exit(-1);
	  }
	}
      }
    source.close();
    m_ndomains = m_ndomX * m_ndomY * m_ndomZ;
  }

  void calc_neighbor_dims()
  {
    // determine neighboring domain IDs
    int maxDomNeighbors = 2 * m_dim;
    m_exchDomIdVec.resize(m_ndomains, std::vector<int>(maxDomNeighbors, -1));

    for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {
      // subdomain indices
      int i = {};
      int j = {};
      int k = {};
      i = domIdx % m_ndomX;
      if (m_dim > 1)  { j = domIdx / m_ndomX; }
      if (m_dim == 2) { k = domIdx / (m_ndomX * m_ndomY); }

      // 1D, 2D, and 3D
      // left boundary
      if (i != 0) { m_exchDomIdVec[domIdx][0] = domIdx - 1; }

      // right boundary
      if (i != (m_ndomX - 1)) {
	// ordering change for 1D vs. 2D/3D faces
	if (m_dim == 1) { m_exchDomIdVec[domIdx][1] = domIdx + 1; }
	else { m_exchDomIdVec[domIdx][2] = domIdx + 1; }
      }

      // 2D and 3D
      if (m_dim > 1) {
	// front boundary
	if (j != (m_ndomY - 1)) { m_exchDomIdVec[domIdx][1] = domIdx + m_ndomX; }
	// back boundary
	if (j != 0) { m_exchDomIdVec[domIdx][3] = domIdx - m_ndomX;	}
      }

      // 3D
      if (m_dim > 2) {
	// bottom boundary
	if (k != 0) { m_exchDomIdVec[domIdx][4] = domIdx - (m_ndomX * m_ndomY); }
	// top boundary
	if (k != (m_ndomZ - 1)) { m_exchDomIdVec[domIdx][5] = domIdx + (m_ndomX * m_ndomY); }
      }
    }
  }

private:
  int m_dim = {};
  int m_ndomX = {};
  int m_ndomY = {};
  int m_ndomZ = {};
  int m_overlap = {};
  int m_ndomains = {};

  std::vector<std::vector<int>> m_exchDomIdVec;
};



template<class MeshType, class AppType>
class SubdomainFrizzi
{

public:
  using app_t      = AppType;
  using graph_t    = typename AppType::mesh_connectivity_graph_type;
  using state_t    = typename AppType::state_type;
  using jacob_t    = typename AppType::jacobian_type;

  using stepper_t  =
    decltype(pressio::ode::create_implicit_stepper(pressio::ode::StepScheme(),
						   declval<AppType&>()) );

  using lin_solver_tag = pressio::linearsolvers::iterative::Bicgstab;
  using linsolver_t    = pressio::linearsolvers::Solver<lin_solver_tag, jacob_t>;
  using nonlinsolver_t =
    decltype( pressio::create_newton_solver( std::declval<stepper_t &>(),
					     std::declval<linsolver_t&>()) );

public:
  SubdomainFrizzi(const int domainIndex,
		  const MeshType & mesh,
		  const std::array<int, 3> & meshFullDim,
		  BCType bcLeft, BCType bcFront,
		  BCType bcRight, BCType bcBack,
		  pressiodemoapps::Euler2d probId,
		  pressio::ode::StepScheme odeScheme,
		  pressiodemoapps::InviscidFluxReconstruction order,
		  const int icflag)
    : m_domIdx(domainIndex)
    , m_dims(meshFullDim)
    , m_mesh(&mesh)
    , m_app(std::make_shared<AppType>(pressiodemoapps::create_problem_eigen
				      (mesh, probId, order,
				       BCFunctor(bcLeft),  BCFunctor(bcFront),
				       BCFunctor(bcRight), BCFunctor(bcBack),
				       icflag)))
    , m_stepper(pressio::ode::create_implicit_stepper(odeScheme, *m_app))
    , m_linSolverObj(std::make_shared<linsolver_t>())
    , m_nonlinSolver(pressio::create_newton_solver(m_stepper, *m_linSolverObj))
    , m_state(m_app->initialCondition())
  {
    if (order != pressiodemoapps::InviscidFluxReconstruction::FirstOrder){
      std::runtime_error("SubdomainFrizzi: inviscid reconstruction must be first oder");
    }

    m_nonlinSolver.setStopTolerance(1e-5);
    init_bc_state(probId, order);
  }

  void allocateStorageForHistory(const int count){
    for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
      // createState creates a new state with all elements equal to zero
      m_stateHistVec.emplace_back(m_app->createState());
    }
  }

  int nx() const{ return m_dims[0]; }
  int ny() const{ return m_dims[1]; }
  int nz() const{ return m_dims[2]; }

private:
  void init_bc_state(pressiodemoapps::Euler2d probId,
		     pressiodemoapps::InviscidFluxReconstruction order)
  {
    // // TODO: can presumably remove this when routines generalized to unstructured format
    const int bcStencilSize = (pressiodemoapps::reconstructionTypeToStencilSize(order) - 1) / 2;
    const int bcStencilDof    = bcStencilSize * AppType::numDofPerCell;
    const int numDofStencilBc = 2 * bcStencilDof * (m_dims[0] + m_dims[1] + m_dims[2]);
    pressiodemoapps::resize(m_stateBCs, numDofStencilBc);
    m_stateBCs.fill(0.0);
  }

private:
  const int m_domIdx;
  std::array<int,3> m_dims = {};

public:
  MeshType const * m_mesh;
  std::shared_ptr<AppType> m_app;
  stepper_t m_stepper;
  std::shared_ptr<linsolver_t> m_linSolverObj;
  nonlinsolver_t m_nonlinSolver;

  state_t m_state;
  state_t m_stateBCs;
  std::vector<state_t> m_stateHistVec;
};



template<class AppType, class MeshType, class ...Args>
auto create_subdomains(const std::vector<std::string> & meshPaths,
		       const std::vector<MeshType> & meshes,
		       const Tiling & tiling,
		       pressiodemoapps::Euler2d probId,
		       Args && ... args)
{

  using subdomain_t = SubdomainFrizzi<MeshType, AppType>;
  std::vector<subdomain_t> result;

  const int ndomX = tiling.countX();
  const int ndomY = tiling.countY();
  const int ndomZ = tiling.countZ();
  const int ndomains = tiling.count();

  // the actual BC used are defaulted to Dirichlet, and modified below
  // when they need to be physical BCs
  BCType bcLeft  = BCType::SchwarzDirichlet;
  BCType bcRight = BCType::SchwarzDirichlet;
  BCType bcFront = BCType::SchwarzDirichlet;
  BCType bcBack  = BCType::SchwarzDirichlet;

  // determine boundary conditions for each subdomain, specify app type
  for (int domIdx = 0; domIdx < ndomains; ++domIdx)
    {
      std::cout << " domIdx = " << domIdx << std::endl;

      const int i = domIdx % ndomX;
      const int j = domIdx / ndomX;

      // left physical boundary
      if (i==0) {
	bcLeft = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Left);
      }

      // right physical boundary
      if (i == (ndomX - 1)){
	bcRight = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Right);
      }

      // front physical boundary
      if (j==0){
	bcFront = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Front);
      }

      // back physical boundary
      if (j == (ndomY - 1)){
	bcBack = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Back);
      }

      // mesh dimensions (non-hyper-reduced only)
      const auto meshFullDims = read_mesh_dims(meshPaths[domIdx], domIdx);

      result.emplace_back(subdomain_t(domIdx, meshes[domIdx], meshFullDims,
				      bcLeft, bcFront, bcRight, bcBack,
				      probId, std::forward<Args>(args)...)
			  );
      std::cout << " create_subdomains finished " << std::endl;
    }

  return result;
}


template<class SubdomainType>
class SchwarzDecompFrizzi
{
  using app_t   = typename SubdomainType::app_t;
  using graph_t = typename app_t::mesh_connectivity_graph_type;
  using state_t = typename app_t::state_type;

public:
  SchwarzDecompFrizzi(std::vector<SubdomainType> & subdomains,
		      std::shared_ptr<const Tiling> tiling,
		      std::vector<double> & dtVec)
    : m_dofPerCell(app_t::numDofPerCell)
    , m_tiling(tiling)
    , m_subdomainVec(subdomains)
  {
    std::cout << " **** decomp constructor: start *** \n";
    m_dofPerCell = app_t::numDofPerCell;

    setup_controller(dtVec);
    for (int domIdx = 0; domIdx < subdomains.size(); ++domIdx) {
      m_subdomainVec[domIdx].allocateStorageForHistory(m_controlItersVec[domIdx]);
    }

    // set up communication patterns
    // FIXME: need to move this bcStencilSize somwwhere else
    m_bcStencilSize = 1; ///(pressiodemoapps::reconstructionTypeToStencilSize(order) - 1) / 2;
    // check_mesh_compat(); // a little error checking
    calc_exch_graph(m_bcStencilSize);

    // first communication
    for (int domIdx = 0; domIdx < subdomains.size(); ++domIdx) {
      broadcast_bcState(domIdx);
    }
    // set up ghost filling graph
    calc_ghost_graph();

    std::cout << " **** decomp constructor: complete *** \n";
  }

  void calc_controller_step(int outerStep, double time, const double rel_err_tol,
			    const double abs_err_tol, const int convergeStepMax)
  {
    namespace pode = pressio::ode;

    const auto ndomains = m_subdomainVec.size();

    // store initial step for resetting if Schwarz iter does not converge
    for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
      m_subdomainVec[domIdx].m_stateHistVec[0] = m_subdomainVec[domIdx].m_state;
    }

    int convergeStep = 0;
    std::vector<std::array<double, 2>> convergeVals(ndomains);
    while (convergeStep < convergeStepMax) {
      std::cout << "Schwarz iteration " << convergeStep + 1 << std::endl;

      for (int domIdx = 0; domIdx < ndomains; ++domIdx)
      {
	auto & currSubdomain = m_subdomainVec[domIdx];

	// reset to beginning of controller time
	auto timeDom = time;
	auto stepDom = outerStep * m_controlItersVec[domIdx];
	const auto dtDom = m_dt[domIdx];
	const auto dtWrap = pode::StepSize<double>(dtDom);
	// controller inner loop
	for (int innerStep = 0; innerStep < m_controlItersVec[domIdx]; ++innerStep) {
	  const auto startTimeWrap = pode::StepStartAt<double>(timeDom);
	  const auto stepWrap = pode::StepCount(stepDom);
	  currSubdomain.m_stepper(currSubdomain.m_state, startTimeWrap,
					 stepWrap, dtWrap, currSubdomain.m_nonlinSolver);

	  // for last iteration, compute convergence criteria
	  // important to do this before saving history,
	  // as stateHistVec still has last convergence loop's state
	  if (innerStep == (m_controlItersVec[domIdx] - 1)) {
	    convergeVals[domIdx] = calcConvergence(currSubdomain.m_state,
						   currSubdomain.m_stateHistVec.back());
	  }
	  // store intra-step history
	  currSubdomain.m_stateHistVec[innerStep + 1] = currSubdomain.m_state;
	  // set (interpolated) boundary conditions
	  // update local step and time
	  stepDom++;
	  timeDom += dtDom;
	} // domain loop

	// broadcast boundary conditions
	broadcast_bcState(domIdx);
      }

      // check convergence for all domains, break if conditions met
      double abs_err = 0.0;
      double rel_err = 0.0;
      for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
	abs_err += convergeVals[domIdx][0];
	rel_err += convergeVals[domIdx][1];
      }
      abs_err /= ndomains;
      rel_err /= ndomains;
      cout << "Average abs err: " << abs_err << endl;
      cout << "Average rel err: " << rel_err << endl;
      if ((rel_err < rel_err_tol) || (abs_err < abs_err_tol)) {
	break;
      }

      convergeStep++;

      // reset interior state if not converged
      for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
	m_subdomainVec[domIdx].m_state = m_subdomainVec[domIdx].m_stateHistVec[0];
      }
    } // convergence loop
  }

private:
  template<class state_t>
  void comm_stateBc(const int startIdx,
		    const int endIdx,
		    const graph_t & exchGraph,
		    state_t & bcState,
		    const state_t & intState)
  {
    int exchCellIdx;
    for (int bcCellIdx = startIdx; bcCellIdx < endIdx; ++bcCellIdx) {
      for (int stencilIdx = 0; stencilIdx < m_bcStencilSize; ++stencilIdx) {
	exchCellIdx = exchGraph(bcCellIdx, stencilIdx);
	for (int dof = 0; dof < m_dofPerCell; ++dof) {
	  bcState((bcCellIdx + stencilIdx) * m_dofPerCell + dof) = intState(exchCellIdx * m_dofPerCell + dof);
	}
      }
    }
  }

  void calc_ghost_graph()
  {
    const auto & tiling = *m_tiling;
    const auto & exchDomIdVec = tiling.exchDomIdVec();

    m_ghostGraphVec.resize(tiling.count());
    for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

      const auto & meshObj = *(m_subdomainVec[domIdx].m_mesh);
      const auto intGraph = meshObj.graph();
      const int nx = m_subdomainVec[domIdx].nx();
      const int ny = m_subdomainVec[domIdx].ny();
      // int nz = m_subdomainVec[domIdx].nz();

      const auto & rowsBd = meshObj.graphRowsOfCellsNearBd();
      pda::resize(m_ghostGraphVec[domIdx], int(rowsBd.size()), 2 * tiling.dim());
      m_ghostGraphVec[domIdx].fill(-1);

      for (decltype(rowsBd.size()) it = 0; it < rowsBd.size(); ++it) {

	// ASK FR: is there an instance when rowsBd[it] != intGraph(rowsBd[it], 0)?
	//    The indices appear to be identical
	// TODO: this is all totally wrong for higher order

	const auto smPt = rowsBd[it];
	const auto left0  = intGraph(smPt, 1);
	const auto front0 = intGraph(smPt, 2);
	const auto right0 = intGraph(smPt, 3);
	const auto back0  = intGraph(smPt, 4);

	// int stencilIdx = 0; // first order
	const int rowIdx = smPt / nx;
	const int colIdx = smPt % nx;
	int bcCellIdx;
	if (left0 == -1) {
	  if (exchDomIdVec[domIdx][0] != -1) {
	    bcCellIdx = rowIdx;
	    m_ghostGraphVec[domIdx](it, 0) = bcCellIdx * m_dofPerCell;
	  }
	}

	if (front0 == -1) {
	  if (exchDomIdVec[domIdx][1] != -1) {
	    bcCellIdx = ny + colIdx;
	    m_ghostGraphVec[domIdx](it, 1) = bcCellIdx * m_dofPerCell;
	  }
	}

	if (right0 == -1) {
	  if (exchDomIdVec[domIdx][2] != -1) {
	    bcCellIdx = ny + nx + rowIdx;
	    m_ghostGraphVec[domIdx](it, 2) = bcCellIdx * m_dofPerCell;
	  }
	}

	if (back0 == -1) {
	  if (exchDomIdVec[domIdx][3] != -1) {
	    bcCellIdx = 2 * ny + nx + colIdx;
	    m_ghostGraphVec[domIdx](it, 3) = bcCellIdx * m_dofPerCell;
	  }
	}
	// TODO: extend to higher order, 3D

      } // boundary cell loop
    } // domain loop
  }

  void broadcast_bcState(const int domIdx)
  {
    const auto & tiling = *m_tiling;
    const auto & exchDomIdVec = tiling.exchDomIdVec();
    const auto* domState = &m_subdomainVec[domIdx].m_state;

    int startIdx, endIdx;
    for (int neighIdx = 0; neighIdx < (int) exchDomIdVec[domIdx].size(); ++neighIdx) {

      int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
      if (neighDomIdx == -1) {
	continue;  // not a Schwarz BC
      }

      int nxNeigh = m_subdomainVec[neighDomIdx].nx();
      int nyNeigh = m_subdomainVec[neighDomIdx].ny();
      auto* neighStateBCs = &m_subdomainVec[neighDomIdx].m_stateBCs;
      const auto & neighExchGraph = m_exchGraphVec[neighDomIdx];

      // TODO: extend to 3D, need to change L/R and F/B indices to account for nzNeigh

      // this domain is the neighboring domain's left neighbor
      if (neighIdx == 2) {
	startIdx = 0;
	endIdx = nyNeigh;
	comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
      }

      // this domain is the neighboring domain's front neighbor
      if (neighIdx == 3) {
	startIdx = nyNeigh;
	endIdx = nyNeigh + nxNeigh;
	comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
      }

      // this domain is the neighboring domain's right neighbor
      if (neighIdx == 0) {
	startIdx = nyNeigh + nxNeigh;
	endIdx = 2 * nyNeigh + nxNeigh;
	comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
      }

      // this domain is the neighboring domain's back neighbor
      if (neighIdx == 1) {
	startIdx = 2 * nyNeigh + nxNeigh;
	endIdx = 2 * nyNeigh + 2 * nxNeigh;
	comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
      }

      // this domain is the neighboring domain's bottom neighbor
      // if (neighIdx == 5) {
      //   startIdx = ;
      //   endIdx = ;
      // }
      // this domain is the neighboring domain's top neighbor
      // if (neighIdx == 4) {
      //   startIdx = ;
      //   endIdx = ;
      // }
      // comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBc, domState);
    }
  }

  void calc_exch_graph(const int bcStencil)
  {
    const auto & tiling = *m_tiling;

    // TODO: extend to 3D

    // BC cell indexing example
    // L/R is from bottom to top, F/B is from left to right
    // Trying to mix cell ordering and face ordering
    //                  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    //                 ¦_(4, 1)_¦_(5, 1)_¦_(6, 1)_¦_(7, 1)_¦_(8, 1)_¦
    //  _ _ _ _ _ _ _ _¦_(4, 0)_¦_(5, 0)_¦_(6, 0)_¦_(7, 0)_¦_(8, 0)_¦_ _ _ _ _ _ _ _ _
    // ¦_(3,1)_¦_(3,0)_|________|________|________|________|________|_(12,1)_¦_(12,0)_|
    // ¦_(2,1)_¦_(2,0)_|________|________|________|________|________|_(11,1)_¦_(11,0)_|
    // ¦_(1,1)_¦_(1,0)_|________|________|________|________|________|_(10,1)_¦_(10,0)_|
    // ¦_(0,1)_¦_(0,0)_|________|________|________|________|________|_(9, 1)_¦_(9, 0)_|
    //                 ¦_(13,0)_¦_(14,0)_¦_(15,0)_¦_(16,0)_¦_(17,0)_¦
    //                 ¦_(13,1)_¦_(14,1)_¦_(15,1)_¦_(16,1)_¦_(17,1)_¦

    m_exchGraphVec.resize(tiling.count());

    const auto overlap = tiling.overlap();
    const auto exchDomIds = tiling.exchDomIdVec();
    for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

      // this domain's mesh and dimensions
      int nx = m_subdomainVec[domIdx].nx();
      int ny = m_subdomainVec[domIdx].ny();

      // TODO: generalize to 3D
      pda::resize(m_exchGraphVec[domIdx], 2*nx + 2*ny, bcStencil);
      m_exchGraphVec[domIdx].fill(-1);

      // loop through neighboring domains
      for (int neighIdx = 0; neighIdx < (int) exchDomIds[domIdx].size(); ++neighIdx) {

	int neighDomIdx = exchDomIds[domIdx][neighIdx];
	if (neighDomIdx == -1) {
	  continue;  // not a Schwarz BC
	}

	// neighboring domain mesh and dimensions
	int nxNeigh = m_subdomainVec[neighDomIdx].nx();
	int nyNeigh = m_subdomainVec[neighDomIdx].ny();

	int exchCellIdx;

	// east-west neighbors will share a row index
	// left
	if (neighIdx == 0) {
	  int bcCellIdx = 0; // left boundary is the start
	  for (int yIdx = 0; yIdx < ny; ++yIdx) {
	    for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
	      exchCellIdx = (nxNeigh * (yIdx + 1) - 1) - overlap - stencilIdx;
	      m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
	    }
	    bcCellIdx++;
	  }
	}

	// north-south neighbors will share a column index
	// "front"
	if (neighIdx == 1) {
	    // skip left boundary indices
	  int bcCellIdx = ny * bcStencil;
	  for (int xIdx = 0; xIdx < nx; ++xIdx) {
	    for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
	      exchCellIdx = (overlap + stencilIdx) * nxNeigh + xIdx;
	      m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
	    }
	    bcCellIdx++;
	  }
	}

	// right
	if (neighIdx == 2) {
	  // skip left and "front" boundary indices
	  int bcCellIdx = (ny + nx) * bcStencil;
	  for (int yIdx = 0; yIdx < ny; ++yIdx) {
	    for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
	      exchCellIdx = nxNeigh * yIdx + overlap + stencilIdx;
	      // TODO: check for a different problem
	      m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
	    }
	    bcCellIdx++;
	  }
	}

	// "back"
	if (neighIdx == 3) {
	  // skip left, "front", and right boundary indices
	  int bcCellIdx = (2*ny + nx) * bcStencil;
	  for (int xIdx = 0; xIdx < nx; ++xIdx) {
	    for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
	      exchCellIdx = (nyNeigh - 1 - overlap - stencilIdx) * nxNeigh + xIdx;
	      m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
	    }
	    bcCellIdx++;
	  }
	}

	// TODO: generalize to 3D

      } // neighbor loop
    } // domain loop
  }

  void setup_controller(std::vector<double> & dtVec){
    const auto & tiling = *m_tiling;

    // physical time step checks
    m_dt = dtVec;
    if (m_dt.size() == 1) {
      m_dt.resize(tiling.count(), m_dt[0]);
    } else {
      if (m_dt.size() != (size_t) tiling.count()) {
	cerr << "dt.size() must be 1 or ndomains, exiting" << endl;
	exit(-1);
      }
    }
    m_dtMax = *std::max_element(m_dt.begin(), m_dt.end());

    // controller time step checks
    m_controlItersVec.resize(tiling.count());
    for (int domIdx = 0; domIdx < (int) m_dt.size(); ++domIdx) {
      double niters = m_dtMax / m_dt[domIdx];
      if (round(niters) == niters) {
	m_controlItersVec[domIdx] = int(round(niters));
      } else {
	cerr << "dt of domain " << domIdx << " (" << m_dt[domIdx]
	     << ") is not an integer divisor of maximum dt ("
	     << m_dtMax << ")" << endl;
	exit(-1);
      }
    }
  }

  template <class state_t>
  auto calcConvergence(const state_t & state1, const state_t & state2)
  {
    // TODO: assumed to be an Eigen state, not sure how to generalize
    // TODO: compute convergence for each variable separately

    int numDOF = state1.size();
    if (state2.size() != numDOF) {
      std::cerr << "state1 size does not match state2 size, "
		<< numDOF
		<< " vs. " << state2.size()
		<< std::endl;
      exit(-1);
    }
    // absolute error
    double abs_err = (state1 - state2).squaredNorm();
    // handle edge cases for relative error
    double rel_err;
    double basenorm = state1.squaredNorm();
    if (basenorm > 0) { rel_err = abs_err / basenorm; }
    else {
      if (abs_err > 0) { rel_err = 1.0; }
      else { rel_err = 0.0; }
    }
    std::array<double, 2> errArr = {abs_err, rel_err};
    return errArr;
  }

public:
  int m_dofPerCell;
  std::shared_ptr<const Tiling> m_tiling;
  std::vector<SubdomainType> & m_subdomainVec;
  int m_bcStencilSize;
  double m_dtMax;
  std::vector<double> m_dt;
  std::vector<graph_t> m_exchGraphVec;
  std::vector<int> m_controlItersVec;
  std::vector<graph_t> m_ghostGraphVec;
};

}//end namespace pdaschwarz

#endif
