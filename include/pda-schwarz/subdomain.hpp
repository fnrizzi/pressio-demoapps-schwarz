
#ifndef PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_

#include <string>
#include <vector>

#include "pressio/ode_steppers_implicit.hpp"

#include "./tiling.hpp"
#include "./custom_bcs.hpp"


using namespace std;

namespace pdaschwarz {

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;
namespace pls = pressio::linearsolvers;

template<class prob_t, class mesh_t, class app_type>
class Subdomain
{

public:

    using app_t      = app_type;
    using graph_t    = typename app_t::mesh_connectivity_graph_type;
    using state_t    = typename app_t::state_type;
    using jacob_t    = typename app_t::jacobian_type;

    using stepper_t  =
        decltype(pressio::ode::create_implicit_stepper(pressio::ode::StepScheme(),
            declval<app_t&>())
        );

    using lin_solver_tag = pressio::linearsolvers::iterative::Bicgstab;
    using linsolver_t    = pressio::linearsolvers::Solver<lin_solver_tag, jacob_t>;
    using nonlinsolver_t =
        decltype( pressio::create_newton_solver( declval<stepper_t &>(),
                            declval<linsolver_t&>()) );

public:

    Subdomain(const int domainIndex,
        const mesh_t & mesh,
        const array<int, 3> & meshFullDim,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag)
    : m_domIdx(domainIndex)
    , m_dims(meshFullDim)
    , m_mesh(&mesh)
    , m_app(make_shared<app_t>(pressiodemoapps::create_problem_eigen(
            mesh, probId, order,
            BCFunctor<mesh_t>(bcLeft),  BCFunctor<mesh_t>(bcFront),
            BCFunctor<mesh_t>(bcRight), BCFunctor<mesh_t>(bcBack),
            icflag)))
    , m_stepper(pressio::ode::create_implicit_stepper(odeScheme, *m_app))
    , m_linSolverObj(make_shared<linsolver_t>())
    , m_nonlinSolver(pressio::create_newton_solver(m_stepper, *m_linSolverObj))
    , m_state(m_app->initialCondition())
    {
        if (order != pressiodemoapps::InviscidFluxReconstruction::FirstOrder){
            runtime_error("Subdomain: inviscid reconstruction must be first oder");
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

    void init_bc_state(prob_t probId,
        pressiodemoapps::InviscidFluxReconstruction order)
    {
        // TODO: can presumably remove this when routines generalized to unstructured format
        const int bcStencilSize = (pressiodemoapps::reconstructionTypeToStencilSize(order) - 1) / 2;
        const int bcStencilDof = bcStencilSize * app_t::numDofPerCell;
        const int numDofStencilBc = 2 * bcStencilDof * (m_dims[0] + m_dims[1] + m_dims[2]);
        pressiodemoapps::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
    }

private:

    const int m_domIdx;
    array<int,3> m_dims = {};

public:

    mesh_t const * m_mesh;
    shared_ptr<app_t> m_app;
    stepper_t m_stepper;
    shared_ptr<linsolver_t> m_linSolverObj;
    nonlinsolver_t m_nonlinSolver;

    state_t m_state;
    state_t m_stateBCs;
    vector<state_t> m_stateHistVec;

};

//
// auxiliary function to create a vector of meshes given a count and meshRoot
//
auto create_meshes(string const & meshRoot, const int n)
{
    using mesh_t = pda::cellcentered_uniform_mesh_eigen_type;
    vector<mesh_t> meshes;
    vector<string> meshPaths;
    for (int i = 0; i < n; ++i) {
        meshPaths.emplace_back(meshRoot + "/domain_" + to_string(i));
        meshes.emplace_back( pda::load_cellcentered_uniform_mesh_eigen(meshPaths.back()) );
    }
    return pair(meshPaths, meshes);
}

//
// auxiliary function to read *full* mesh dims from files
//
array<int,3> read_mesh_dims(const string & meshPath, int domIdx)
{
    // Read uniform mesh dimensions, as this was removed from PDA
    // NOTE: this should be made obsolete when code is generalized
    // to connectivity graph, doesn't depend on uniform mesh

    const auto inFile = meshPath + "/info.dat";
    ifstream foundFile(inFile);
    if (!foundFile) {
        cout << "file not found " << inFile << endl;
        exit(EXIT_FAILURE);
    }

    array<int, 3> dims;
    dims.fill(0); // default
    ifstream source(inFile, ios_base::in);
    string line;
    while (getline(source, line)) {
        istringstream ss(line);
        string colVal;
        ss >> colVal;
        if (colVal == "nx") {
            ss >> colVal;
            dims[0] = stoi(colVal);
        }
        else if (colVal == "ny") {
            ss >> colVal;
            dims[1] = stoi(colVal);
        }
        else if (colVal == "nz") {
            ss >> colVal;
            dims[2] = stoi(colVal);
        }
    }
    source.close();
    return dims;
}

template<class app_t, class mesh_t, class prob_t, class ...Args>
auto create_subdomains(const vector<string> & meshPaths,
		               const vector<mesh_t> & meshes,
		               const Tiling & tiling,
		               prob_t probId,
		               Args && ... args)
{

    using subdomain_t = Subdomain<prob_t, mesh_t, app_t>;
    vector<subdomain_t> result;

    const int ndomX = tiling.countX();
    const int ndomY = tiling.countY();
    const int ndomZ = tiling.countZ();
    const int ndomains = tiling.count();

    // determine boundary conditions for each subdomain, specify app type
    for (int domIdx = 0; domIdx < ndomains; ++domIdx)
    {

        // the actual BC used are defaulted to Dirichlet, and modified below
        // when they need to be physical BCs
        BCType bcLeft  = BCType::SchwarzDirichlet;
        BCType bcRight = BCType::SchwarzDirichlet;
        BCType bcFront = BCType::SchwarzDirichlet;
        BCType bcBack  = BCType::SchwarzDirichlet;

        const int i = domIdx % ndomX;
        const int j = domIdx / ndomX;

        // left physical boundary
        if (i == 0) {
            bcLeft = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Left);
        }

        // right physical boundary
        if (i == (ndomX - 1)) {
            bcRight = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Right);
        }

        // back physical boundary
        if (j == 0) {
            bcBack = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Back);
        }

        // front physical boundary
        if (j == (ndomY - 1)) {
            bcFront = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Front);
        }

        // mesh dimensions (non-hyper-reduced only)
        const auto meshFullDims = read_mesh_dims(meshPaths[domIdx], domIdx);

        result.emplace_back(subdomain_t(
            domIdx, meshes[domIdx], meshFullDims,
            bcLeft, bcFront, bcRight, bcBack,
            probId, forward<Args>(args)...)
        );
    }

    return result;
}

}

#endif