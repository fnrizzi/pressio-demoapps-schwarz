

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

template<class mesh_t>
struct BCFunctor
{
    using graph_t  = typename mesh_t::graph_t;
    using scalar_t = typename mesh_t::scalar_t;
    // TODO: not sure if there's a way to template state_t, since app type is templated on BCFunctor (circular?)
    using state_t  = Eigen::Matrix<scalar_t,-1,1>;

    BCType m_bcSwitch;

    // m_stateBcs is shared by EVERY BCFunctor
    // m_graphBcs is a unique mask on m_stateBcs, for each BCFunctor
    // This is because BCFunctor has no internal left/right/front/back reference, and
    //  only understand position in graph from GLOBAL (subdomain) cell index
    state_t* m_stateBcs = nullptr;
    vector<int>* m_graphBcs = nullptr;

    BCFunctor(BCType bcSwitch) : m_bcSwitch(bcSwitch){}

    void setIntPtr(state_t* stateBcs){
        m_stateBcs = stateBcs;
    }

    void setIntPtr(vector<int>* graphBcs){
        m_graphBcs = graphBcs;
    }

    template<class ...Args>
    void operator()(Args && ... args) const
    {
        switch(m_bcSwitch)
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
    void SchwarzDirichletBC(const int gRow, ConnecRowType const & connectivityRow,
                            const double cellX, const double cellY,
                            const StateT & currentState, int numDofPerCell,
                            const double cellWidth, T & ghostValues) const
    {
        // TODO: any way to only check this once? Seems wasteful.
        if ((m_stateBcs == nullptr) || (m_graphBcs == nullptr)) {
            runtime_error("m_stateBcs or m_graphBcs not set");
        }

        const auto bcIndex = (*m_graphBcs)[gRow];
        ghostValues[0] = (*m_stateBcs)(bcIndex+0);
        ghostValues[1] = (*m_stateBcs)(bcIndex+1);
        ghostValues[2] = (*m_stateBcs)(bcIndex+2);
        ghostValues[3] = (*m_stateBcs)(bcIndex+3);

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

auto getPhysBCs(pda::Euler2d probId, pda::impl::GhostRelativeLocation rloc)
{
    // Riemann
    switch(probId)
    {
        case pda::Euler2d::Riemann:
            // All boundaries are homogeneous Neumann
            return BCType::Euler2DHomogNeumann;
            break;

        default:
            throw runtime_error("Invalid probId for getPhysBCs()");

    }

}

}

#endif
