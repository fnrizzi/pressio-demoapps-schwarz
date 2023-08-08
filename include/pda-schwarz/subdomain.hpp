
#ifndef PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_

#include <string>
#include <vector>

#include "pressio/ode_steppers_implicit.hpp"

#include "./custom_bcs.hpp"


using namespace std;

namespace pdaschwarz {

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;
namespace pls  = pressio::linearsolvers;

template<
    class prob_t,
    class order_t,
    class scheme_t,
    class mesh_t,
    class app_t>
class Subdomain
{
    public:

        using graph_t        = typename mesh_t::graph_t;
        using state_t        = typename app_t::state_type;
        using jacob_t        = typename app_t::jacobian_type;
        using stepper_t      = decltype( pode::create_implicit_stepper(declval<scheme_t>(), declval<app_t &>()) );
        // TODO: generalize to more than BICGSTAB
        using linsolver_t    = pls::Solver<pls::iterative::Bicgstab, jacob_t>;
        using nonlinsolver_t = decltype( pressio::create_newton_solver( declval<stepper_t &>(), declval<linsolver_t&>()) );

    public:

        Subdomain(
            prob_t probId,
            order_t order,
            scheme_t scheme,
            const int domainIndex,
            const string meshRootIn,
            BCType bcLeft,
            BCType bcFront,
            BCType bcRight,
            BCType bcBack,
            const int controlItersIn,
            const int icflag)
        :   domIdx(domainIndex),
            meshRoot(meshRootIn),
            mesh(pda::load_cellcentered_uniform_mesh_eigen(meshRoot + "/domain_" + to_string(domIdx))),
            app(pda::create_problem_eigen(mesh, probId, order, BCFunctor(bcLeft), BCFunctor(bcFront), BCFunctor(bcRight), BCFunctor(bcBack), icflag)),
            stepper(pode::create_implicit_stepper(scheme, app)),
            nonlinSolver(pressio::create_newton_solver(stepper, *linSolverObj)),
            controlIters(controlItersIn)
        {

            // mesh dimensions (non-hyper-reduced only)
            // TODO: can presumably remove this when routines generalized to unstructured format
            m_bcStencilSize = (pda::reconstructionTypeToStencilSize(order) - 1) / 2;
            read_mesh_dims();

            // states
            state = app.initialCondition();
            for (int histIdx = 0; histIdx < controlIters + 1; ++histIdx) {
                stateHistVec.emplace_back(app.initialCondition());
            }
            init_bc_state();

            // solver
            nonlinSolver.setStopTolerance(1e-5);

        }

        // getters
        int nx() const{ return m_nx; }
        int ny() const{ return m_ny; }
        int nz() const{ return m_nz; }

    private:
        void read_mesh_dims()
        {
            // Read uniform mesh dimensions, as this was removed from PDA
            // NOTE: this should be made obsolete when code is generalized to connectivity graph, doesn't depend on uniform mesh

            const auto inFile = meshRoot + "/domain_" + to_string(domIdx) + "/info.dat";
            ifstream foundFile(inFile);
            if(!foundFile){
            cout << "file not found " << inFile << endl;
            exit(EXIT_FAILURE);
            }

            ifstream source( inFile, ios_base::in);
            string line;
            while (getline(source, line) )
            {
                istringstream ss(line);
                string colVal;
                ss >> colVal;

                if (colVal == "nx"){
                    ss >> colVal;
                    m_nx = stoi(colVal);
                }

                else if (colVal == "ny"){
                    ss >> colVal;
                    m_ny = stoi(colVal);
                }

                else if (colVal == "nz"){
                    ss >> colVal;
                    m_nz = stoi(colVal);
                }

            }
            source.close();
        }

        void init_bc_state() {

            int bcStencilDof = m_bcStencilSize * app_t::numDofPerCell;
            int numDofStencilBc = 2 * bcStencilDof * (m_nx + m_ny + m_nz);
            pda::resize(stateBCs, numDofStencilBc);
            stateBCs.fill(0.0);

        }

    // TODO: scope correctly

    public:
        state_t state;
        state_t stateBCs;
        vector<state_t> stateHistVec;
        int controlIters;

    private:
        const string meshRoot;
        const int domIdx;

        // mesh dimensions (non-hyper-reduced)
        int m_nx = 0;
        int m_ny = 0;
        int m_nz = 0;
        int m_bcStencilSize;

    public:
        linsolver_t* linSolverObj;
        mesh_t mesh;
        app_t app;
        stepper_t stepper;
        nonlinsolver_t nonlinSolver;

};

}

#endif