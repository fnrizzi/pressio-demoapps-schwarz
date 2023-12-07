
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


namespace pdaschwarz {

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;
namespace pls = pressio::linearsolvers;
namespace prom = pressio::rom;
namespace plspg = pressio::rom::lspg;

template<class mesh_type, class state_type>
class SubdomainBase{
public:
    using state_t = state_type;
    using mesh_t = mesh_type;
    using graph_t = typename mesh_t::graph_t;

    virtual void allocateStorageForHistory(const int) = 0;
    virtual void doStep(pode::StepStartAt<double>, pode::StepCount, pode::StepSize<double>) = 0;
    virtual void storeStateHistory(const int) = 0;
    virtual void resetStateFromHistory() = 0;
    virtual void updateFullState() = 0;
    virtual const mesh_t & getMesh() const = 0;
    virtual const graph_t & getNeighborGraph() const = 0;
    virtual int getDofPerCell() const = 0;
    virtual state_t * getState() = 0;
    virtual state_t * getStateBCs() = 0;
    virtual void setBCPointer(pda::impl::GhostRelativeLocation, state_t * ) = 0;
    virtual void setBCPointer(pda::impl::GhostRelativeLocation, graph_t *) = 0;
    virtual state_t & getLastStateInHistory() = 0;
};


template<class mesh_t, class app_type>
class SubdomainFOM: public SubdomainBase<mesh_t, typename app_type::state_type>
{
public:
    using app_t   = app_type;
    using graph_t = typename mesh_t::graph_t;
    using state_t = typename app_t::state_type;
    using jacob_t = typename app_t::jacobian_type;
    using base_t = SubdomainBase<mesh_t, state_t>;

    using stepper_t  =
        decltype(pressio::ode::create_implicit_stepper(pressio::ode::StepScheme(),
            std::declval<app_t&>())
        );

    using lin_solver_tag = pressio::linearsolvers::iterative::Bicgstab;
    using linsolver_t    = pressio::linearsolvers::Solver<lin_solver_tag, jacob_t>;
    using nonlinsolver_t =
        decltype( pressio::create_newton_solver( std::declval<stepper_t &>(),
                            std::declval<linsolver_t&>()) );

public:
    template<class prob_t>
    SubdomainFOM(
        const int domainIndex,
        const mesh_t & mesh,
        const graph_t & neighborGraph,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams)
    : m_domIdx(domainIndex)
    , m_mesh(&mesh)
    , m_neighborGraph(&neighborGraph)
    , m_app(std::make_shared<app_t>(pressiodemoapps::create_problem_eigen(
            mesh, probId, order,
            BCFunctor<mesh_t>(bcLeft),  BCFunctor<mesh_t>(bcFront),
            BCFunctor<mesh_t>(bcRight), BCFunctor<mesh_t>(bcBack),
            icflag, userParams)))
    , m_state(m_app->initialCondition())
    , m_stepper(pressio::ode::create_implicit_stepper(odeScheme, *(this->m_app)))
    , m_linSolverObj(std::make_shared<linsolver_t>())
    , m_nonlinSolver(pressio::create_newton_solver(m_stepper, *m_linSolverObj))
    {
        init_bc_state();

        m_nonlinSolver.setStopTolerance(1e-5);
    }

    state_t & getLastStateInHistory() final { return m_stateHistVec.back(); }

    void setBCPointer(pda::impl::GhostRelativeLocation grl, state_t * v) final {
        m_app->setBCPointer(grl, v);
    }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, graph_t * v) final {
      m_app->setBCPointer(grl, v);
    }

    state_t * getStateBCs() final { return &m_stateBCs; }
    state_t * getState() final { return &m_state; }
    int getDofPerCell() const final { return m_app->numDofPerCell(); }
    const mesh_t & getMesh() const final { return *m_mesh; }
    const graph_t & getNeighborGraph() const final { return *m_neighborGraph; }

    void init_bc_state()
    {
        // count number of neighbor ghost cells in neighborGraph
        int numGhostCells = 0;
        const auto & rowsBd = m_mesh->graphRowsOfCellsNearBd();
        for (int bdIdx = 0; bdIdx < rowsBd.size(); ++bdIdx) {
            auto rowIdx = rowsBd[bdIdx];
            // start at 1 to ignore own ID
            for (int colIdx = 1; colIdx < m_neighborGraph->cols(); ++colIdx) {
                if ((*m_neighborGraph)(rowIdx, colIdx) != -1) {
                    numGhostCells++;
                }
            }
        }
        const int numDofStencilBc = m_app->numDofPerCell() * numGhostCells;
        pressiodemoapps::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
    }

    void allocateStorageForHistory(const int count) final {
        for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
            // createState creates a new state with all elements equal to zero
            this->m_stateHistVec.emplace_back(this->m_app->createState());
        }
    }

    void doStep(pode::StepStartAt<double> startTime,
        pode::StepCount step,
        pode::StepSize<double> dt) final
    {
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
    int m_domIdx;
    mesh_t const * m_mesh;
    graph_t const * m_neighborGraph;
    std::shared_ptr<app_t> m_app;
    state_t m_state;
    state_t m_stateBCs;
    std::vector<state_t> m_stateHistVec;

    stepper_t m_stepper;
    std::shared_ptr<linsolver_t> m_linSolverObj;
    nonlinsolver_t m_nonlinSolver;

};



template<class mesh_t, class app_type>
class SubdomainROM: public SubdomainBase<mesh_t, typename app_type::state_type>
{
    using app_t    = app_type;
    using graph_t = typename mesh_t::graph_t;
    using scalar_t = typename app_t::scalar_type;
    using state_t  = typename app_t::state_type;
    using base_t = SubdomainBase<mesh_t, state_t>;

    using trans_t = decltype(read_vector_from_binary<scalar_t>(std::declval<std::string>()));
    using basis_t = decltype(read_matrix_from_binary<scalar_t>(std::declval<std::string>(), std::declval<int>()));
    using trial_t = decltype(prom::create_trial_column_subspace<
        state_t>(std::declval<basis_t&&>(), std::declval<trans_t&&>(), true));

public:
    template<class prob_t>
    SubdomainROM(
        const int domainIndex,
        const mesh_t & mesh,
        const graph_t & neighborGraph,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams,
        const std::string & transRoot,
        const std::string & basisRoot,
        int nmodes)
    : m_domIdx(domainIndex)
    , m_mesh(&mesh)
    , m_neighborGraph(&neighborGraph)
    , m_app(std::make_shared<app_t>(pressiodemoapps::create_problem_eigen(
            mesh, probId, order,
            BCFunctor<mesh_t>(bcLeft),  BCFunctor<mesh_t>(bcFront),
            BCFunctor<mesh_t>(bcRight), BCFunctor<mesh_t>(bcBack),
            icflag, userParams)))
    , m_state(m_app->initialCondition())
    , m_nmodes(nmodes)
    , m_trans(read_vector_from_binary<scalar_t>(transRoot + "_" + std::to_string(domainIndex) + ".bin"))
    , m_basis(read_matrix_from_binary<scalar_t>(basisRoot + "_" + std::to_string(domainIndex) + ".bin", nmodes))
    , m_trialSpace(prom::create_trial_column_subspace<state_t>(std::move(m_basis), std::move(m_trans), true))
    , m_stateReduced(m_trialSpace.createReducedState())
    {
        init_bc_state();

        // project initial conditions
        auto u = pressio::ops::clone(this->m_state);
        pressio::ops::update(u, 0., this->m_state, 1, m_trialSpace.translationVector(), -1);
        pressio::ops::product(::pressio::transpose(), 1., m_trialSpace.basis(), u, 0., m_stateReduced);
        m_trialSpace.mapFromReducedState(m_stateReduced, this->m_state);

    }

    state_t & getLastStateInHistory() final { return m_stateHistVec.back(); }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, state_t * v) final{
        m_app->setBCPointer(grl, v);
    }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, graph_t * v) final{
        m_app->setBCPointer(grl, v);
    }

    state_t * getStateBCs() final { return &m_stateBCs; }
    state_t * getState() final { return &m_state; }
    int getDofPerCell() const final { return m_app->numDofPerCell(); }
    const mesh_t & getMesh() const final { return *m_mesh; }
    const graph_t & getNeighborGraph() const final { return *m_neighborGraph; }

    void init_bc_state()
    {
        // count number of neighbor ghost cells in neighborGraph
        int numGhostCells = 0;
        const auto & rowsBd = m_mesh->graphRowsOfCellsNearBd();
        for (int bdIdx = 0; bdIdx < rowsBd.size(); ++bdIdx) {
            auto rowIdx = rowsBd[bdIdx];
            // start at 1 to ignore own ID
            for (int colIdx = 1; colIdx < m_neighborGraph->cols(); ++colIdx) {
                if ((*m_neighborGraph)(rowIdx, colIdx) != -1) {
                    numGhostCells++;
                }
            }
        }
        const int numDofStencilBc = m_app->numDofPerCell() * numGhostCells;
        pressiodemoapps::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
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
    int m_domIdx;
    mesh_t const * m_mesh;
    graph_t const * m_neighborGraph;
    std::shared_ptr<app_t> m_app;
    state_t m_state;
    state_t m_stateBCs;
    std::vector<state_t> m_stateHistVec;

    int m_nmodes;
    trans_t m_trans;
    basis_t m_basis;
    trial_t m_trialSpace;
    state_t m_stateReduced;
    std::vector<state_t> m_stateReducedHistVec;

};

template<class mesh_t, class app_type>
class SubdomainLSPG: public SubdomainROM<mesh_t, app_type>
{

private:
    using app_t    = app_type;
    using graph_t  = typename mesh_t::graph_t;
    using scalar_t = typename app_t::scalar_type;
    using state_t  = typename app_t::state_type;

    using trans_t = decltype(read_vector_from_binary<scalar_t>(std::declval<std::string>()));
    using basis_t = decltype(read_matrix_from_binary<scalar_t>(std::declval<std::string>(), std::declval<int>()));
    using trial_t = decltype(prom::create_trial_column_subspace<
        state_t>(std::declval<basis_t&&>(), std::declval<trans_t&&>(), true));

    using hessian_t   = Eigen::Matrix<scalar_t, -1, -1>; // TODO: generalize?
    using solver_tag  = pressio::linearsolvers::direct::HouseholderQR;
    using linsolver_t = pressio::linearsolvers::Solver<solver_tag, hessian_t>;

    using problem_t       = decltype(plspg::create_unsteady_problem(pressio::ode::StepScheme(), std::declval<trial_t&>(), std::declval<app_t&>()));
    using stepper_t       = decltype(std::declval<problem_t>().lspgStepper());
    using nonlinsolver_t  = decltype(pressio::create_gauss_newton_solver(std::declval<stepper_t&>(), std::declval<linsolver_t&>()));

public:

    template<class prob_t>
    SubdomainLSPG(
        const int domainIndex,
        const mesh_t & mesh,
        const graph_t & neighborGraph,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pressiodemoapps::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams,
        const std::string & transRoot,
        const std::string & basisRoot,
        const int nmodes)
    : SubdomainROM<mesh_t, app_type>::SubdomainROM(
        domainIndex, mesh, neighborGraph,
        bcLeft, bcFront, bcRight, bcBack,
        probId, odeScheme, order, icflag, userParams,
        transRoot, basisRoot, nmodes)
    , m_problem(plspg::create_unsteady_problem(odeScheme, this->m_trialSpace, *(this->m_app)))
    , m_stepper(m_problem.lspgStepper())
    , m_linSolverObj(std::make_shared<linsolver_t>())
    , m_nonlinSolver(pressio::create_gauss_newton_solver(m_stepper, *m_linSolverObj))
    {

    }

    void doStep(pode::StepStartAt<double> startTime, pode::StepCount step, pode::StepSize<double> dt) final {
        m_stepper(this->m_stateReduced, startTime, step, dt, m_nonlinSolver);
    }

private:
    problem_t m_problem;
    stepper_t m_stepper;
    std::shared_ptr<linsolver_t> m_linSolverObj;
    nonlinsolver_t m_nonlinSolver;
};

//
// auxiliary function to create a vector of meshes given a count and meshRoot
//
auto create_meshes(std::string const & meshRoot, const int n)
{
    using mesh_t = pda::cellcentered_uniform_mesh_eigen_type;
    using graph_t = typename mesh_t::graph_t;

    std::vector<mesh_t> meshes;
    std::vector<graph_t> neighborGraphs(n);

    for (int domIdx = 0; domIdx < n; ++domIdx) {
        // read mesh
        std::string meshPath = meshRoot + "/domain_" + std::to_string(domIdx);
        meshes.emplace_back( pda::load_cellcentered_uniform_mesh_eigen(meshPath) );

        // read neighbor connectivity
        const auto numNeighbors = (meshes.back().stencilSize() - 1) * meshes.back().dimensionality();
        const auto graphNumCols = numNeighbors + 1;
        pda::resize(neighborGraphs[domIdx], meshes.back().sampleMeshSize(), graphNumCols);

        // this is ripped directly from mesh_read_connectivity, since it doesn't generalize the file name
        const auto inFile = meshPath + "/connectivity_neighbor.dat";
        std::ifstream foundFile(inFile);
        if (!foundFile) {
            std::cout << "file not found " << inFile << std::endl;
            exit(EXIT_FAILURE);
        }

        std::ifstream source(inFile, std::ios_base::in);
        std::string line;
        std::size_t count = 0;
        while (std::getline(source, line))
        {
            std::istringstream ss(line);
            std::string colVal;
            ss >> colVal;
            neighborGraphs[domIdx](count, 0) = std::stoi(colVal);

            for (auto graphIdx = 1; graphIdx <= graphNumCols - 1; ++graphIdx) {
                ss >> colVal;
                neighborGraphs[domIdx](count, graphIdx) = std::stoi(colVal);
            }
            count++;
        }
        source.close();
    }

    return std::pair(meshes, neighborGraphs);
}

//
// all domains are assumed to be FOM domains
//
template<class app_t, class mesh_t, class graph_t, class prob_t>
auto create_subdomains(
    const std::vector<mesh_t> & meshes,
    const std::vector<graph_t> & neighborGraphs,
    const Tiling & tiling,
    prob_t probId,
    pode::StepScheme odeScheme,
    pressiodemoapps::InviscidFluxReconstruction order,
    int icFlag = 0,
    const std::unordered_map<std::string, typename app_t::scalar_type> & userParams = {})
{
    auto ndomains = tiling.count();
    std::vector<std::string> domFlagVec(ndomains, "FOM");
    std::vector<int> nmodesVec(ndomains, -1);

    return create_subdomains<app_t>(meshes, neighborGraphs, tiling,
        probId, odeScheme, order,
        domFlagVec, "", "", nmodesVec, icFlag, userParams);

}

//
// Subdomain type specified by domFlagVec
//
template<class app_t, class mesh_t, class graph_t, class prob_t>
auto create_subdomains(
    const std::vector<mesh_t> & meshes,
    const std::vector<graph_t> & neighborGraphs,
    const Tiling & tiling,
    prob_t probId,
    pode::StepScheme odeScheme,
    pressiodemoapps::InviscidFluxReconstruction order,
    const std::vector<std::string> & domFlagVec,
    const std::string & transRoot,
    const std::string & basisRoot,
    const std::vector<int> & nmodesVec,
    int icFlag = 0,
    const std::unordered_map<std::string, typename app_t::scalar_type> & userParams = {})
{

    // add checks that vectors are all same size?

    // using subdomain_t = Subdomain<prob_t, mesh_t, app_t>;
    using subdomain_t = SubdomainBase<mesh_t, typename app_t::state_type>;
    std::vector<std::shared_ptr<subdomain_t>> result;

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

        if (domFlagVec[domIdx] == "FOM") {
            result.emplace_back(std::make_shared<SubdomainFOM<mesh_t, app_t>>(
                domIdx, meshes[domIdx], neighborGraphs[domIdx],
                bcLeft, bcFront, bcRight, bcBack,
                probId, odeScheme, order, icFlag, userParams));
        }
        else if (domFlagVec[domIdx] == "LSPG") {
            result.emplace_back(std::make_shared<SubdomainLSPG<mesh_t, app_t>>(
                domIdx, meshes[domIdx], neighborGraphs[domIdx],
                bcLeft, bcFront, bcRight, bcBack,
                probId, odeScheme, order, icFlag, userParams,
                transRoot, basisRoot, nmodesVec[domIdx]));
        }
        else {
            std::runtime_error("Invalid subdomain flag value: " + domFlagVec[domIdx]);
        }
    }

    return result;
}

}

#endif
