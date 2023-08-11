

#ifndef PRESSIODEMOAPPS_SCHWARZ_CUSTOMBCS_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_CUSTOMBCS_HPP_

#include "pressiodemoapps/impl/ghost_relative_locations.hpp"
#include "pressiodemoapps/euler2d.hpp"
#include "pressiodemoapps/swe2d.hpp"

using namespace std;

namespace pda = pressiodemoapps;

namespace pdaschwarz{

enum class BCType {
    Euler2DHomogNeumann,
    SchwarzDirichlet,
};

struct BCFunctor
{
    BCType bcSwitch;

    BCFunctor(BCType bcSwitchIn) : bcSwitch(bcSwitchIn){}

    template<class ...Args>
    void operator()(Args && ... args) const
    {
        switch(bcSwitch)
        {
            case BCType::Euler2DHomogNeumann:
                Euler2DHomogNeumannBC(forward<Args>(args)...);
		        break;
            case BCType::SchwarzDirichlet:
                SchwarzDirichletBC(forward<Args>(args)...);
		        break;
            default:
                throw runtime_error("Invalid probId for getPhysBCs()");
        };
    }

private:

    /*=========================
        PHYSICAL BOUNDARIES
    =========================*/

    template<class ConnecRowType, class StateT, class T>
    void Euler2DHomogNeumannBC(const int /*unused*/, ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               const StateT & currentState, int numDofPerCell,
                               const double cellWidth, T & ghostValues) const
    {
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID*numDofPerCell;
        ghostValues[0] = currentState(uIndex);
        ghostValues[1] = currentState(uIndex+1);
        ghostValues[2] = currentState(uIndex+2);
        ghostValues[3] = currentState(uIndex+3);
    }

    template<class ConnecRowType, class FactorsType>
    void Euler2DHomogNeumannBC(ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        factorsForBCJac = {1.,1.,1.,1.};
    }

    /*=========================
        SCHWARZ BOUNDARIES
    =========================*/

    template<class ConnecRowType, class StateT, class T>
    void SchwarzDirichletBC(const int /*unused*/, ConnecRowType const & connectivityRow,
                            const double cellX, const double cellY,
                            const StateT & currentState, int numDofPerCell,
                            const double cellWidth, T & ghostValues) const
    {
        runtime_error("Schwarz BC not implemented");
    }

    template<class ConnecRowType, class FactorsType>
    void SchwarzDirichletBC(ConnecRowType const & connectivityRow,
            const double cellX, const double cellY,
            int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        factorsForBCJac = {0.,0.,0.,0.};
    }

};

/*============================
    DEFAULT SPECIFICATIONS
=============================*/

// 2D Euler
auto getPhysBCs(pda::Euler2d probId, pda::impl::GhostRelativeLocation rloc)
{
    // Riemann
    switch(probId)
    {
        case pda::Euler2d::Riemann:{
            // All boundaries are homogeneous Neumann
            return BCType::Euler2DHomogNeumann;
        }

        default:
            throw runtime_error("Invalid probId for getPhysBCs()");

    }

}

}

#endif
