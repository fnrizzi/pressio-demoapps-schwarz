
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
    private:

        using graph_t        = typename mesh_t::graph_t;
        using state_t        = typename app_t::state_type;
        using jacob_t        = typename app_t::jacobian_type;
        using stepper_t      = decltype( pode::create_implicit_stepper(declval<scheme_t>(), declval<app_t &>()) );
        // // TODO: generalize to more than BICGSTAB
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

            // states
            state = app.initialCondition();
            for (int histIdx = 0; histIdx < controlIters + 1; ++histIdx) {
                stateHistVec.emplace_back(app.initialCondition());
            }

            // solver
            nonlinSolver.setStopTolerance(1e-5);

            // mesh dimensions (non-hyper-reduced only)
            // TODO: can presumably remove this when routines generalized to unstructured format
            get_mesh_dims();

        }


    private:
        void get_mesh_dims()
        {
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
                    nx = stoi(colVal);
                }

                else if (colVal == "ny"){
                    ss >> colVal;
                    ny = stoi(colVal);
                }

                else if (colVal == "nz"){
                    ss >> colVal;
                    nz = stoi(colVal);
                }

            }
            source.close();
        }


    public:
        state_t state;
        vector<state_t> stateHistVec;
        int controlIters;

        // mesh dimensions (non-hyper-reduced)
        int nx = 0;
        int ny = 0;
        int nz = 0;

    private:
        const string meshRoot;
        const int domIdx;

    // TODO: scope correctly
    public:
        linsolver_t* linSolverObj;
        mesh_t mesh;
        app_t app;
        stepper_t stepper;
        nonlinsolver_t nonlinSolver;

};

}

#endif