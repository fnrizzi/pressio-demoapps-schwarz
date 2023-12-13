
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
    virtual const mesh_t & getMeshStencil() const = 0;
    virtual const mesh_t & getMeshFull() const = 0;
    virtual const graph_t & getNeighborGraph() const = 0;
    virtual int getDofPerCell() const = 0;
    virtual state_t * getStateStencil() = 0;
    virtual state_t * getStateFull() = 0;
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
        pda::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams)
    : m_domIdx(domainIndex)
    , m_mesh(&mesh)
    , m_neighborGraph(&neighborGraph)
    , m_app(std::make_shared<app_t>(pda::create_problem_eigen(
            mesh, probId, order,
            BCFunctor<mesh_t>(bcLeft),  BCFunctor<mesh_t>(bcFront),
            BCFunctor<mesh_t>(bcRight), BCFunctor<mesh_t>(bcBack),
            icflag, userParams)))
    , m_state(m_app->initialCondition())
    , m_stepper(pressio::ode::create_implicit_stepper(odeScheme, *(m_app)))
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
    state_t * getStateStencil() final { return &m_state; }
    state_t * getStateFull() final { return &m_state; }
    int getDofPerCell() const final { return m_app->numDofPerCell(); }
    const mesh_t & getMeshStencil() const final { return *m_mesh; }
    const mesh_t & getMeshFull() const final { return *m_mesh; }
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
        pda::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
    }

    void allocateStorageForHistory(const int count) final {
        for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
            // createState creates a new state with all elements equal to zero
            m_stateHistVec.emplace_back(m_app->createState());
        }
    }

    void doStep(pode::StepStartAt<double> startTime,
        pode::StepCount step,
        pode::StepSize<double> dt) final
    {
        m_stepper(m_state, startTime, step, dt, m_nonlinSolver);
    }

    void storeStateHistory(const int step) final {
        m_stateHistVec[step] = m_state;
    }

    void resetStateFromHistory() final {
        m_state = m_stateHistVec[0];
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
        pda::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams,
        const std::string & transRoot,
        const std::string & basisRoot,
        int nmodes)
    : m_domIdx(domainIndex)
    , m_mesh(&mesh)
    , m_neighborGraph(&neighborGraph)
    , m_app(std::make_shared<app_t>(pda::create_problem_eigen(
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
        auto u = pressio::ops::clone(m_state);
        pressio::ops::update(u, 0., m_state, 1, m_trialSpace.translationVector(), -1);
        pressio::ops::product(::pressio::transpose(), 1., m_trialSpace.basis(), u, 0., m_stateReduced);
        m_trialSpace.mapFromReducedState(m_stateReduced, m_state);

    }

    state_t & getLastStateInHistory() final { return m_stateHistVec.back(); }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, state_t * v) final{
        m_app->setBCPointer(grl, v);
    }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, graph_t * v) final{
        m_app->setBCPointer(grl, v);
    }

    state_t * getStateBCs() final { return &m_stateBCs; }
    state_t * getStateStencil() final { return &m_state; }
    state_t * getStateFull() final { return &m_state; }
    int getDofPerCell() const final { return m_app->numDofPerCell(); }
    const mesh_t & getMeshStencil() const final { return *m_mesh; }
    const mesh_t & getMeshFull() const final { return *m_mesh; }
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
        pda::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
    }

    void allocateStorageForHistory(const int count){
        for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
            m_stateHistVec.emplace_back(m_app->createState());
            m_stateReducedHistVec.emplace_back(m_trialSpace.createReducedState());
        }
    }

    void storeStateHistory(const int step) final {
        m_stateHistVec[step] = m_state;
        m_stateReducedHistVec[step] = m_stateReduced;
    }

    void resetStateFromHistory() final {
        m_state = m_stateHistVec[0];
        m_stateReduced = m_stateReducedHistVec[0];
    }

    void updateFullState() final {
        m_trialSpace.mapFromReducedState(m_stateReduced, m_state);
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
        pda::InviscidFluxReconstruction order,
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


template<class mesh_t, class app_type>
class SubdomainHyper: public SubdomainBase<mesh_t, typename app_type::state_type>
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

    using stencil_t  = decltype(create_cell_gids_vector_and_fill_from_ascii(std::declval<std::string>()));
    using transHyp_t = decltype(reduce_vector_on_stencil_mesh(std::declval<trans_t&>(), std::declval<stencil_t&>(), std::declval<int>()));
    using basisHyp_t = decltype(reduce_matrix_on_stencil_mesh(std::declval<basis_t&>(), std::declval<stencil_t&>(), std::declval<int>()));
    using trialHyp_t = decltype(prom::create_trial_column_subspace<
        state_t>(std::declval<basisHyp_t&&>(), std::declval<transHyp_t&&>(), true));

public:

    template<class prob_t>
    SubdomainHyper(
        const int domainIndex,
        const mesh_t & meshFull,
        const graph_t & neighborGraph,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pda::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams,
        const std::string & transRoot,
        const std::string & basisRoot,
        const int nmodes,
        const mesh_t & meshHyper,
        const std::string & meshPathHyper)
    : m_domIdx(domainIndex)
    , m_meshFull(&meshFull)
    , m_meshHyper(&meshHyper)
    , m_appFull(std::make_shared<app_t>(pda::create_problem_eigen(
            meshFull, probId, order,
            BCFunctor<mesh_t>(bcLeft),  BCFunctor<mesh_t>(bcFront),
            BCFunctor<mesh_t>(bcRight), BCFunctor<mesh_t>(bcBack),
            icflag, userParams)))
    , m_appHyper(std::make_shared<app_t>(pda::create_problem_eigen(
            meshHyper, probId, order,
            BCFunctor<mesh_t>(bcLeft),  BCFunctor<mesh_t>(bcFront),
            BCFunctor<mesh_t>(bcRight), BCFunctor<mesh_t>(bcBack),
            icflag, userParams)))
    , m_neighborGraph(&neighborGraph)
    , m_stencilFile(meshPathHyper + "/stencil_mesh_gids.dat")
    , m_sampleFile(meshPathHyper + "/sample_mesh_gids.dat")
    , m_stencilGids(create_cell_gids_vector_and_fill_from_ascii(m_stencilFile))
    , m_nmodes(nmodes)
    , m_transFull(read_vector_from_binary<scalar_t>(transRoot + "_" + std::to_string(domainIndex) + ".bin"))
    , m_basisFull(read_matrix_from_binary<scalar_t>(basisRoot + "_" + std::to_string(domainIndex) + ".bin", nmodes))
    , m_transRead(read_vector_from_binary<scalar_t>(transRoot + "_" + std::to_string(domainIndex) + ".bin"))
    , m_basisRead(read_matrix_from_binary<scalar_t>(basisRoot + "_" + std::to_string(domainIndex) + ".bin", nmodes))
    , m_transHyper(reduce_vector_on_stencil_mesh(m_transRead, m_stencilGids, m_appFull->numDofPerCell()))
    , m_basisHyper(reduce_matrix_on_stencil_mesh(m_basisRead, m_stencilGids, m_appFull->numDofPerCell()))
    , m_trialSpaceFull(prom::create_trial_column_subspace<
       state_t>(std::move(m_basisFull), std::move(m_transFull), true))
    , m_trialSpaceHyper(prom::create_trial_column_subspace<
       state_t>(std::move(m_basisHyper), std::move(m_transHyper), true))
    {

        m_stateStencil = m_appHyper->initialCondition();
        m_stateFull = m_appFull->initialCondition();
        m_stateReduced = m_trialSpaceHyper.createReducedState();
        init_bc_state();

        // project initial conditions
        auto u = pressio::ops::clone(m_stateFull);
        pressio::ops::update(u, 0., m_stateFull, 1, m_trialSpaceFull.translationVector(), -1);
        pressio::ops::product(::pressio::transpose(), 1., m_trialSpaceFull.basis(), u, 0., m_stateReduced);
        m_trialSpaceFull.mapFromReducedState(m_stateReduced, m_stateFull);

    }

    state_t & getLastStateInHistory() final { return m_stateHistVec.back(); }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, state_t * v) final{
        m_appHyper->setBCPointer(grl, v);
    }
    void setBCPointer(pda::impl::GhostRelativeLocation grl, graph_t * v) final{
        m_appHyper->setBCPointer(grl, v);
    }

    state_t * getStateBCs() final { return &m_stateBCs; }
    state_t * getStateStencil() final { return &m_stateStencil; }
    state_t * getStateFull() final {
        m_trialSpaceFull.mapFromReducedState(m_stateReduced, m_stateFull);
        return &m_stateFull;
    }
    int getDofPerCell() const final { return m_appHyper->numDofPerCell(); }
    const mesh_t & getMeshStencil() const final { return *m_meshHyper; }
    const mesh_t & getMeshFull() const final { return *m_meshFull; }
    const graph_t & getNeighborGraph() const final { return *m_neighborGraph; }

    void init_bc_state()
    {

        // count number of neighbor ghost cells in neighborGraph
        int numGhostCells = 0;
        const auto & rowsBd = m_meshHyper->graphRowsOfCellsNearBd();
        for (int bdIdx = 0; bdIdx < rowsBd.size(); ++bdIdx) {
            auto rowIdx = rowsBd[bdIdx];
            // start at 1 to ignore own ID
            for (int colIdx = 1; colIdx < m_neighborGraph->cols(); ++colIdx) {
                if ((*m_neighborGraph)(rowIdx, colIdx) != -1) {
                    numGhostCells++;
                }
            }
        }
        const int numDofStencilBc = m_appHyper->numDofPerCell() * numGhostCells;
        pda::resize(m_stateBCs, numDofStencilBc);
        m_stateBCs.fill(0.0);
    }

    void allocateStorageForHistory(const int count){
        for (int histIdx = 0; histIdx < count + 1; ++histIdx) {
            m_stateHistVec.emplace_back(m_appHyper->createState());
            m_stateReducedHistVec.emplace_back(m_trialSpaceHyper.createReducedState());
        }
    }

    void storeStateHistory(const int step) final {
        m_stateHistVec[step] = m_stateStencil;
        m_stateReducedHistVec[step] = m_stateReduced;
    }

    void resetStateFromHistory() final {
        m_stateStencil = m_stateHistVec[0];
        m_stateReduced = m_stateReducedHistVec[0];
    }

    void updateFullState() final {
        m_trialSpaceHyper.mapFromReducedState(m_stateReduced, m_stateStencil);
    }

public:
    int m_domIdx;
    mesh_t const * m_meshFull;
    mesh_t const * m_meshHyper;
    graph_t const * m_neighborGraph;
    std::shared_ptr<app_t> m_appFull;
    std::shared_ptr<app_t> m_appHyper;

    state_t m_stateStencil;  // on stencil mesh
    state_t m_stateFull;     // on full, unsampled mesh (required for projection)
    state_t m_stateReduced;  // latent state
    state_t m_stateBCs;
    std::vector<state_t> m_stateHistVec;
    std::vector<state_t> m_stateReducedHistVec;

    std::string m_stencilFile;
    std::string m_sampleFile;
    stencil_t m_stencilGids;

    int m_nmodes;
    trans_t m_transFull;
    basis_t m_basisFull;
    trans_t m_transRead;
    basis_t m_basisRead;
    trial_t m_trialSpaceFull;
    transHyp_t m_transHyper;
    basisHyp_t m_basisHyper;
    trialHyp_t m_trialSpaceHyper;

};

template<class mesh_t, class app_type>
class SubdomainLSPGHyper: public SubdomainHyper<mesh_t, app_type>
{

private:
    using app_t    = app_type;
    using graph_t  = typename mesh_t::graph_t;
    using scalar_t = typename app_t::scalar_type;
    using state_t  = typename app_t::state_type;

    using trans_t = decltype(read_vector_from_binary<scalar_t>(std::declval<std::string>()));
    using basis_t = decltype(read_matrix_from_binary<scalar_t>(std::declval<std::string>(), std::declval<int>()));
    using stencil_t  = decltype(create_cell_gids_vector_and_fill_from_ascii(std::declval<std::string>()));
    using transHyp_t = decltype(reduce_vector_on_stencil_mesh(std::declval<trans_t&>(), std::declval<stencil_t&>(), 1));
    using basisHyp_t = decltype(reduce_matrix_on_stencil_mesh(std::declval<basis_t&>(), std::declval<stencil_t&>(), 1));
    using trialHyp_t = decltype(prom::create_trial_column_subspace<
        state_t>(std::declval<basisHyp_t&&>(), std::declval<transHyp_t&&>(), true));

    using hessian_t   = Eigen::Matrix<scalar_t, -1, -1>; // TODO: generalize?
    using solver_tag  = pressio::linearsolvers::direct::HouseholderQR;
    using linsolver_t = pressio::linearsolvers::Solver<solver_tag, hessian_t>;

    using updaterHyp_t   = HypRedUpdater<scalar_t>;
    using problemHyp_t   = decltype(plspg::create_unsteady_problem(pressio::ode::StepScheme(), std::declval<trialHyp_t&>(), std::declval<app_t&>(), std::declval<updaterHyp_t&>()));
    using stepperHyp_t   = decltype(std::declval<problemHyp_t>().lspgStepper());
    using nonlinsolverHyp_t = decltype(pressio::create_gauss_newton_solver(std::declval<stepperHyp_t&>(), std::declval<linsolver_t&>()));

public:

    template<class prob_t>
    SubdomainLSPGHyper(
        const int domainIndex,
        const mesh_t & meshFull,
        const graph_t & neighborGraph,
        BCType bcLeft, BCType bcFront,
        BCType bcRight, BCType bcBack,
        prob_t probId,
        pressio::ode::StepScheme odeScheme,
        pda::InviscidFluxReconstruction order,
        const int icflag,
        const std::unordered_map<std::string, typename mesh_t::scalar_type> & userParams,
        const std::string & transRoot,
        const std::string & basisRoot,
        const int nmodes,
        const mesh_t & meshHyper,
        const std::string & meshPathHyper)
    : SubdomainHyper<mesh_t, app_type>::SubdomainHyper(
        domainIndex, meshFull, neighborGraph,
        bcLeft, bcFront, bcRight, bcBack,
        probId, odeScheme, order, icflag, userParams,
        transRoot, basisRoot, nmodes,
        meshHyper, meshPathHyper)
    , m_updaterHyper(create_hyper_updater<mesh_t>(this->getDofPerCell(), this->m_stencilFile, this->m_sampleFile))
    , m_problemHyper(plspg::create_unsteady_problem(odeScheme, this->m_trialSpaceHyper, *(this->m_appHyper), m_updaterHyper))
    , m_stepperHyper(m_problemHyper.lspgStepper())
    , m_linSolverObjHyper(std::make_shared<linsolver_t>())
    , m_nonlinSolverHyper(pressio::create_gauss_newton_solver(m_stepperHyper, *m_linSolverObjHyper))
    {

    }

    void doStep(pode::StepStartAt<double> startTime, pode::StepCount step, pode::StepSize<double> dt) final {
        m_stepperHyper(this->m_stateReduced, startTime, step, dt, m_nonlinSolverHyper);
    }

// TODO: to protected
public:

    updaterHyp_t m_updaterHyper;
    problemHyp_t m_problemHyper;
    stepperHyp_t m_stepperHyper;
    std::shared_ptr<linsolver_t> m_linSolverObjHyper;
    nonlinsolverHyp_t m_nonlinSolverHyper;

};

//
// auxiliary function to create a vector of meshes given a count and meshRoot
//
auto create_meshes(std::string const & meshRoot, const int n)
{
    using mesh_t = pda::cellcentered_uniform_mesh_eigen_type;
    using graph_t = typename mesh_t::graph_t;

    std::vector<mesh_t> meshes;
    std::vector<std::string> meshPaths;
    std::vector<graph_t> neighborGraphs(n);

    for (int domIdx = 0; domIdx < n; ++domIdx) {
        // read mesh
        meshPaths.emplace_back(meshRoot + "/domain_" + std::to_string(domIdx));
        meshes.emplace_back( pda::load_cellcentered_uniform_mesh_eigen(meshPaths.back()) );

        // read neighbor connectivity
        const auto numNeighbors = (meshes.back().stencilSize() - 1) * meshes.back().dimensionality();
        const auto graphNumCols = numNeighbors + 1;
        pda::resize(neighborGraphs[domIdx], meshes.back().sampleMeshSize(), graphNumCols);

        // this is ripped directly from mesh_read_connectivity, since it doesn't generalize the file name
        const auto inFile = meshPaths.back() + "/connectivity_neighbor.dat";
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

    return std::tuple(meshes, meshPaths, neighborGraphs);
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
    pda::InviscidFluxReconstruction order,
    int icFlag = 0,
    const std::unordered_map<std::string, typename app_t::scalar_type> & userParams = {})
{
    auto ndomains = tiling.count();
    std::vector<std::string> domFlagVec(ndomains, "FOM");

    // dummy arguments
    std::vector<int> nmodesVec(ndomains, -1);
    std::vector<mesh_t> meshesHyper(ndomains);
    std::vector<std::string> meshPathsHyper(ndomains, "");

    return create_subdomains<app_t>(meshes, neighborGraphs, tiling,
        probId, odeScheme, order,
        domFlagVec, "", "", nmodesVec,
        icFlag, meshesHyper, meshPathsHyper, userParams);

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
    pda::InviscidFluxReconstruction order,
    const std::vector<std::string> & domFlagVec,
    const std::string & transRoot,
    const std::string & basisRoot,
    const std::vector<int> & nmodesVec,
    int icFlag = 0,
    const std::vector<mesh_t> & meshesHyper = {},
    const std::vector<std::string> & meshPathsHyper = {},
    const std::unordered_map<std::string, typename app_t::scalar_type> & userParams = {})
{

    // add checks that vectors are all same size?

    // using subdomain_t = Subdomain<prob_t, mesh_t, app_t>;
    using subdomain_t = SubdomainBase<mesh_t, typename app_t::state_type>;
    std::vector<std::shared_ptr<subdomain_t>> result;

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
        else if (domFlagVec[domIdx] == "LSPGHyper") {
            // TODO: the access of meshesHyper is a little wonky,
            //      as it's not guaranteed that every mesh be a sample mesh
            result.emplace_back(std::make_shared<SubdomainLSPGHyper<mesh_t, app_t>>(
                domIdx, meshes[domIdx], neighborGraphs[domIdx],
                bcLeft, bcFront, bcRight, bcBack,
                probId, odeScheme, order, icFlag, userParams,
                transRoot, basisRoot, nmodesVec[domIdx],
                meshesHyper[domIdx], meshPathsHyper[domIdx]));
        }
        else {
            std::runtime_error("Invalid subdomain flag value: " + domFlagVec[domIdx]);
        }
    }

    return result;
}

}

#endif
