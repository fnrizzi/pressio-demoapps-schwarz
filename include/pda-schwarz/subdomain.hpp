
#ifndef PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_SUBDOMAIN_HPP_

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
        // // TODO: generalize to more than BICGSTAB
        using linsolver_t    = pls::Solver<pls::iterative::Bicgstab, jacob_t>;
        using nonlinsolver_t = decltype( pressio::create_newton_solver( declval<stepper_t &>(), declval<linsolver_t&>()) );

    public:

        Subdomain(
            prob_t probId,
            order_t order,
            scheme_t scheme,
            mesh_t meshIn,
            BCType bcLeft,
            BCType bcFront,
            BCType bcRight,
            BCType bcBack,
            const int controlItersIn,
            const int icflag)
        :   mesh(meshIn),
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

            nonlinSolver.setStopTolerance(1e-5);
        }

    public:
        state_t state;
        vector<state_t> stateHistVec;
        int controlIters;

    public:
        linsolver_t* linSolverObj;
        mesh_t mesh;
        app_t app;
        stepper_t stepper;
        nonlinsolver_t nonlinSolver;

};

}

#endif