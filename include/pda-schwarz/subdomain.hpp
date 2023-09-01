
#ifndef PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_

#include <string>
#include <vector>

#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/rom_subspaces.hpp"
#include "pressio/rom_lspg_unsteady.hpp"

#include "./tiling.hpp"
#include "./custom_bcs.hpp"
#include "./rom_utils.hpp"


using namespace std;

namespace pdaschwarz {

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;
namespace pls = pressio::linearsolvers;
namespace prom = pressio::rom;
namespace plspg = pressio::rom::lspg;

template<class mesh_t, class app_type>
class SubdomainBase
{

public:

    using app_t      = app_type;
    using state_t    = typename app_t::state_type;

protected:

    template<class prob_t>
    SubdomainBase(const int domainIndex,
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
    , m_state(m_app->initialCondition())
    {
        if (order != pressiodemoapps::InviscidFluxReconstruction::FirstOrder){
            runtime_error("Subdomain: inviscid reconstruction must be first oder");
        }

        init_bc_state(order);
    }

public:

    int nx() const{ return m_dims[0]; }
    int ny() const{ return m_dims[1]; }
    int nz() const{ return m_dims[2]; }

    // we set equal to zero to make these purely virtual
    virtual void allocateStorageForHistory(const int) = 0;
    virtual void doStep(pode::StepStartAt<double>, pode::StepCount, pode::StepSize<double>) = 0;
    virtual void storeStateHistory(const int) = 0;
    virtual void resetStateFromHistory() = 0;
    virtual void updateFullState() = 0;

private:

    void init_bc_state(pressiodemoapps::InviscidFluxReconstruction order)
    {
        // TODO: can presumably remove this when routines generalized to unstructured format
        const int bcStencilSize = (pressiodemoapps::reconstructionTypeToStencilSize(order) - 1) / 2;
        const int bcStencilDof = bcStencilSize * m_app->numDofPerCell();
        const int numDofStencilBc = 2 * bcStencilDof * (m_dims[0] + m_dims[1] + m_dims[2]);
        pressiodemoapps::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
    }

protected:

    int m_domIdx;
    array<int,3> m_dims = {};

public:

    mesh_t const * m_mesh;
    shared_ptr<app_t> m_app;
    state_t m_state;
    state_t m_stateBCs;
    vector<state_t> m_stateHistVec;

};

template<class mesh_t, class app_type>
class SubdomainFOM: public SubdomainBase<mesh_t, app_type>
{

public:

    using app_t   = app_type;
    using state_t = typename app_t::state_type;
    using jacob_t = typename app_t::jacobian_type;

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

    template<class prob_t>
    SubdomainFOM(const int domainIndex,
        const mesh_t & mesh,
        const array<int, 3> & meshFullDim,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag)
    : SubdomainBase<mesh_t, app_t>::SubdomainBase(domainIndex, mesh, meshFullDim, bcLeft, bcFront, bcRight, bcBack, probId, odeScheme, order, icflag)
    , m_stepper(pressio::ode::create_implicit_stepper(odeScheme, *(this->m_app)))
    , m_linSolverObj(make_shared<linsolver_t>())
    , m_nonlinSolver(pressio::create_newton_solver(m_stepper, *m_linSolverObj))
    {
        m_nonlinSolver.setStopTolerance(1e-5);
    }

    void allocateStorageForHistory(const int count) final {
        for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
            // createState creates a new state with all elements equal to zero
            this->m_stateHistVec.emplace_back(this->m_app->createState());
        }
    }

    void doStep(pode::StepStartAt<double> startTime, pode::StepCount step, pode::StepSize<double> dt) final {
        m_stepper(this->m_state, startTime, step, dt, m_nonlinSolver);
    }

    void storeStateHistory(const int step) final {
        this->m_stateHistVec[step] = this->m_state;
    }

    void resetStateFromHistory() final {
        this->m_state = this->m_stateHistVec[0];
    }

    void updateFullState() final {
        // noop
    }

public:

    stepper_t m_stepper;
    shared_ptr<linsolver_t> m_linSolverObj;
    nonlinsolver_t m_nonlinSolver;

};

template<class mesh_t, class app_type>
class SubdomainROM: public SubdomainBase<mesh_t, app_type>
{

    using app_t    = app_type;
    using scalar_t = typename app_t::scalar_type;
    using state_t  = typename app_t::state_type;

    using trans_t = decltype(read_vector_from_binary<scalar_t>(declval<string>()));
    using basis_t = decltype(read_matrix_from_binary<scalar_t>(declval<string>(), declval<int>()));
    using trial_t = decltype(prom::create_trial_column_subspace<
        state_t>(declval<basis_t&&>(), declval<trans_t&&>(), true));

public:

    template<class prob_t>
    SubdomainROM(const int domainIndex,
        const mesh_t & mesh,
        const array<int, 3> & meshFullDim,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag,
        const string & transRoot,
        const string & basisRoot,
        int nmodes)
    : SubdomainBase<mesh_t, app_t>::SubdomainBase(domainIndex, mesh, meshFullDim, bcLeft, bcFront, bcRight, bcBack, probId, odeScheme, order, icflag)
    , m_nmodes(nmodes)
    , m_trans(read_vector_from_binary<scalar_t>(transRoot + "_" + to_string(domainIndex) + ".bin"))
    , m_basis(read_matrix_from_binary<scalar_t>(basisRoot + "_" + to_string(domainIndex) + ".bin", nmodes))
    , m_trialSpace(prom::create_trial_column_subspace<state_t>(move(m_basis), move(m_trans), true))
    , m_stateReduced(m_trialSpace.createReducedState())
    {
        // project initial conditions
        auto u = pressio::ops::clone(this->m_state);
        pressio::ops::update(u, 0., this->m_state, 1, m_trialSpace.translationVector(), -1);
        pressio::ops::product(::pressio::transpose(), 1., m_trialSpace.basis(), u, 0., m_stateReduced);
        m_trialSpace.mapFromReducedState(m_stateReduced, this->m_state);

    }

    void allocateStorageForHistory(const int count){
        for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
            this->m_stateHistVec.emplace_back(this->m_app->createState());
            m_stateReducedHistVec.emplace_back(m_trialSpace.createReducedState());
        }
    }

    void storeStateHistory(const int step) final {
        this->m_stateHistVec[step] = this->m_state;
        this->m_stateReducedHistVec[step] = this->m_stateReduced;
    }

    void resetStateFromHistory() final {
        this->m_state = this->m_stateHistVec[0];
        this->m_stateReduced = this->m_stateReducedHistVec[0];
    }

    void updateFullState() final {
        m_trialSpace.mapFromReducedState(m_stateReduced, this->m_state);
    }

protected:

    int m_nmodes;
    trans_t m_trans;
    basis_t m_basis;
    trial_t m_trialSpace;
    state_t m_stateReduced;
    vector<state_t> m_stateReducedHistVec;

};

template<class mesh_t, class app_type>
class SubdomainLSPG: public SubdomainROM<mesh_t, app_type>
{

private:
    using app_t    = app_type;
    using scalar_t = typename app_t::scalar_type;
    using state_t  = typename app_t::state_type;

    using trans_t = decltype(read_vector_from_binary<scalar_t>(declval<string>()));
    using basis_t = decltype(read_matrix_from_binary<scalar_t>(declval<string>(), declval<int>()));
    using trial_t = decltype(prom::create_trial_column_subspace<
        state_t>(declval<basis_t&&>(), declval<trans_t&&>(), true));

    using hessian_t   = Eigen::Matrix<scalar_t, -1, -1>; // TODO: generalize?
    using solver_tag  = pressio::linearsolvers::direct::HouseholderQR;
    using linsolver_t = pressio::linearsolvers::Solver<solver_tag, hessian_t>;

    using problem_t       = decltype(plspg::create_unsteady_problem(pressio::ode::StepScheme(), declval<trial_t&>(), declval<app_t&>()));
    using stepper_t       = decltype(declval<problem_t>().lspgStepper());
    using nonlinsolver_t  = decltype(pressio::create_gauss_newton_solver(declval<stepper_t&>(), declval<linsolver_t&>()));

public:

    template<class prob_t>
    SubdomainLSPG(const int domainIndex,
        const mesh_t & mesh,
        const array<int, 3> & meshFullDim,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag,
        const string & transRoot,
        const string & basisRoot,
        const int nmodes)
    : SubdomainROM<mesh_t, app_type>::SubdomainROM(domainIndex, mesh, meshFullDim,
        bcLeft, bcFront, bcRight, bcBack,
        probId, odeScheme, order, icflag,
        transRoot, basisRoot, nmodes)
    , m_problem(plspg::create_unsteady_problem(odeScheme, this->m_trialSpace, *(this->m_app)))
    , m_stepper(m_problem.lspgStepper())
    , m_linSolverObj(make_shared<linsolver_t>())
    , m_nonlinSolver(pressio::create_gauss_newton_solver(m_stepper, *m_linSolverObj))
    {

    }

    void doStep(pode::StepStartAt<double> startTime, pode::StepCount step, pode::StepSize<double> dt) final {
        m_stepper(this->m_stateReduced, startTime, step, dt, m_nonlinSolver);
    }

private:

    problem_t m_problem;
    stepper_t m_stepper;
    shared_ptr<linsolver_t> m_linSolverObj;
    nonlinsolver_t m_nonlinSolver;
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

//
// all domains are assumed to be FOM domains
//
template<class app_t, class mesh_t, class prob_t>
auto create_subdomains(const vector<string> & meshPaths,
                       const vector<mesh_t> & meshes,
                       const Tiling & tiling,
                       prob_t probId,
                       pode::StepScheme odeScheme,
                       pressiodemoapps::InviscidFluxReconstruction order,
                       int icFlag = 0)
{
    auto ndomains = tiling.count();
    vector<string> domFlagVec(ndomains, "FOM");
    vector<int> nmodesVec(ndomains, -1);

    return create_subdomains<app_t>(meshPaths, meshes, tiling,
        probId, odeScheme, order,
        domFlagVec, "", "", nmodesVec, icFlag);

}

//
// Subdomain type specified by domFlagVec
//
template<class app_t, class mesh_t, class prob_t>
auto create_subdomains(const vector<string> & meshPaths,
                       const vector<mesh_t> & meshes,
                       const Tiling & tiling,
                       prob_t probId,
                       pode::StepScheme odeScheme,
                       pressiodemoapps::InviscidFluxReconstruction order,
                       const vector<string> & domFlagVec,
                       const string & transRoot,
                       const string & basisRoot,
                       const vector<int> & nmodesVec,
                       int icFlag = 0)
{

    // add checks that vectors are all same size?

    // using subdomain_t = Subdomain<prob_t, mesh_t, app_t>;
    using subdomain_t = SubdomainBase<mesh_t, app_t>;
    vector<shared_ptr<subdomain_t>> result;

    const int ndomX = tiling.countX();
    const int ndomY = tiling.countY();
    // const int ndomZ = tiling.countZ();
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

        if (domFlagVec[domIdx] == "FOM") {
            result.emplace_back(make_shared<SubdomainFOM<mesh_t, app_t>>(domIdx, meshes[domIdx], meshFullDims,
                bcLeft, bcFront, bcRight, bcBack,
                probId, odeScheme, order, icFlag));
        }
        else if (domFlagVec[domIdx] == "LSPG") {
            result.emplace_back(make_shared<SubdomainLSPG<mesh_t, app_t>>(domIdx, meshes[domIdx], meshFullDims,
                bcLeft, bcFront, bcRight, bcBack,
                probId, odeScheme, order, icFlag,
                transRoot, basisRoot, nmodesVec[domIdx]));
        }
        else {
            runtime_error("Invalid subdomain flag value: " + domFlagVec[domIdx]);
        }
    }

    return result;
}

}

#endif