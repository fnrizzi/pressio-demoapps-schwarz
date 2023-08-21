

#ifndef PRESSIODEMOAPPS_SCHWARZ_CUSTOMBCS_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_CUSTOMBCS_HPP_

#include "pressiodemoapps/impl/ghost_relative_locations.hpp"
#include "pressiodemoapps/euler2d.hpp"
#include "pressiodemoapps/swe2d.hpp"

using namespace std;

namespace pda = pressiodemoapps;

namespace pdaschwarz{

enum class BCType {
    HomogNeumann,
    SlipWallVert,
    SlipWallHoriz,
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

    void setInternalPtr(state_t* stateBcs){
        m_stateBcs = stateBcs;
    }

    void setInternalPtr(vector<int>* graphBcs){
        m_graphBcs = graphBcs;
    }

    template<class ...Args>
    void operator()(Args && ... args) const
    {
        switch(m_bcSwitch)
        {
            case BCType::HomogNeumann:
                HomogNeumannBC(forward<Args>(args)...);
		        break;
            case BCType::SchwarzDirichlet:
                SchwarzDirichletBC(forward<Args>(args)...);
		        break;
            case BCType::SlipWallVert:
                SlipWallVertBC(forward<Args>(args)...);
                break;
            case BCType::SlipWallHoriz:
                SlipWallHorizBC(forward<Args>(args)...);
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
    void HomogNeumannBC(const int /*unused*/, ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               const StateT & currentState, int numDofPerCell,
                               const double cellWidth, T & ghostValues) const
    {
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID*numDofPerCell;
        for (int i = 0; i < numDofPerCell; ++i) {
            ghostValues[i] = currentState(uIndex+i);
        }
    }

    template<class ConnecRowType, class FactorsType>
    void HomogNeumannBC(ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        for (int i = 0; i < numDofPerCell; ++i) {
            factorsForBCJac[i] = 1.0;
        }
    }

    template<class ConnecRowType, class StateT, class T>
    void SlipWallVertBC(const int /*unused*/, ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               const StateT & currentState, int numDofPerCell,
                               const double cellWidth, T & ghostValues) const
    {
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID*numDofPerCell;
        for (int i = 0; i < numDofPerCell; ++i) {
            ghostValues[i] = currentState(uIndex+i);
        }
        ghostValues[1] *= -1.0; // reverse x-momentum
    }

    template<class ConnecRowType, class FactorsType>
    void SlipWallVertBC(ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        for (int i = 0; i < numDofPerCell; ++i) {
            factorsForBCJac[i] = 1.0;
        }
        factorsForBCJac[1] = -1.0;
    }

    template<class ConnecRowType, class StateT, class T>
    void SlipWallHorizBC(const int /*unused*/, ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               const StateT & currentState, int numDofPerCell,
                               const double cellWidth, T & ghostValues) const
    {
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID*numDofPerCell;
        for (int i = 0; i < numDofPerCell; ++i) {
            ghostValues[i] = currentState(uIndex+i);
        }
        ghostValues[2] *= -1.0; // reverse y-momentum
    }

    template<class ConnecRowType, class FactorsType>
    void SlipWallHorizBC(ConnecRowType const & connectivityRow,
                               const double cellX, const double cellY,
                               int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        for (int i = 0; i < numDofPerCell; ++i) {
            factorsForBCJac[i] = 1.0;
        }
        factorsForBCJac[2] = -1.0;
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
        // assumes that T can be indexed by [], which is true for demoapps (Eigen::Matrix.row())
        for (int i = 0; i < numDofPerCell; ++i) {
            ghostValues[i] = (*m_stateBcs)(bcIndex+i);
        }

    }

    template<class ConnecRowType, class FactorsType>
    void SchwarzDirichletBC(ConnecRowType const & connectivityRow,
            const double cellX, const double cellY,
            int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        // assumes that FactorsType can be indexed by [], which is true for demoapps (std::array)
        for (int i = 0; i < numDofPerCell; ++i) {
            factorsForBCJac[i] = 0.0;
        }
    }

};

/*============================
    DEFAULT SPECIFICATIONS
=============================*/

auto getPhysBCs(pda::Euler2d probId, pda::impl::GhostRelativeLocation rloc)
{

    switch(probId)
    {
        case pda::Euler2d::Riemann:
            // All boundaries are homogeneous Neumann
            return BCType::HomogNeumann;
            break;

        default:
            throw runtime_error("Invalid probId for getPhysBCs()");

    }
}

auto getPhysBCs(pda::Swe2d probId, pda::impl::GhostRelativeLocation rloc)
{

    switch(probId)
    {
        // SWE is really only SlipWall for now, but PDA doesn't permit Swe2d::SlipWall with custom BCs
        // Just use CustomBCs as a proxy for the normal slip wall case
        case pda::Swe2d::CustomBCs:
            if ((rloc == pda::impl::GhostRelativeLocation::Left) || (rloc == pda::impl::GhostRelativeLocation::Right)) {
                return BCType::SlipWallVert;
            }
            else if ((rloc == pda::impl::GhostRelativeLocation::Front) || (rloc == pda::impl::GhostRelativeLocation::Back)) {
                return BCType::SlipWallHoriz;
            }
            else {
                throw runtime_error("Unexpected GhostRelativeLocation");
            }
            break;

        default:
            throw runtime_error("Invalid probId for getPhysBCs()");

    }
}

}

#endif
