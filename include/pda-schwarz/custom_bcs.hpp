

#ifndef PRESSIODEMOAPPS_SCHWARZ_CUSTOMBCS_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_CUSTOMBCS_HPP_

#include "pressiodemoapps/impl/ghost_relative_locations.hpp"
#include "pressiodemoapps/euler2d.hpp"
#include "pressiodemoapps/swe2d.hpp"


namespace pdaschwarz{

namespace pda = pressiodemoapps;

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
    graph_t* m_graphBcs = nullptr;

    BCFunctor(BCType bcSwitch) : m_bcSwitch(bcSwitch){}

    void setInternalPtr(state_t* stateBcs){
        m_stateBcs = stateBcs;
    }

    void setInternalPtr(graph_t* graphBcs){
        m_graphBcs = graphBcs;
    }

    template<class ...Args>
    void operator()(Args && ... args) const
    {
        switch(m_bcSwitch)
        {
            case BCType::HomogNeumann:
                HomogNeumannBC(std::forward<Args>(args)...);
		        break;
            case BCType::SchwarzDirichlet:
                SchwarzDirichletBC(std::forward<Args>(args)...);
		        break;
            case BCType::SlipWallVert:
                SlipWallVertBC(std::forward<Args>(args)...);
                break;
            case BCType::SlipWallHoriz:
                SlipWallHorizBC(std::forward<Args>(args)...);
                break;
            default:
	      throw std::runtime_error("Invalid probId for getPhysBCs()");
        };
    }

private:

    /*=========================
        PHYSICAL BOUNDARIES
    =========================*/

    template<class ConnecRowType, class StateT, class T>
    void HomogNeumannBC(
        const int /*unused*/, ConnecRowType const & connectivityRow,
        const double cellX, const double cellY,
        const StateT & currentState, int numDofPerCell,
        const double cellWidth, T & ghostValues) const
    {
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID * numDofPerCell;
        for (int i = 0; i < numDofPerCell; ++i) {
            ghostValues[i] = currentState(uIndex+i);
        }
    }

    template<class ConnecRowType, class FactorsType>
    void HomogNeumannBC(
        ConnecRowType const & connectivityRow,
        const double cellX, const double cellY,
        int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        for (int i = 0; i < numDofPerCell; ++i) {
            factorsForBCJac[i] = 1.0;
        }
    }

    template<class ConnecRowType, class StateT, class T>
    void SlipWallVertBC(
        const int /*unused*/, ConnecRowType const & connectivityRow,
        const double cellX, const double cellY,
        const StateT & currentState, int numDofPerCell,
        const double cellWidth, T & ghostValues) const
    {
        // TODO: generalize to 1D/3D

        // this operates under the assumption that this cell does not have ghost cells in two parallel walls
        int stencilSize1D = ghostValues.cols() / numDofPerCell;
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID * numDofPerCell;

        const auto left0  = connectivityRow[1];
        const auto right0  = connectivityRow[3];
        if ((left0 == -1) && (right0 == -1)) {
            throw std::runtime_error("Should not have walls to left and right of same cell");
        }

        if ((left0 == -1) || (right0 == -1)) {
            for (int i = 0; i < numDofPerCell; ++i) {
                ghostValues[i] = currentState(uIndex+i);
            }
            ghostValues[1] *= -1.0; // reverse x-momentum
        }

        // TODO: extend to WENO5
        if (stencilSize1D > 1) {
            const auto left1  = connectivityRow[5];
            const auto right1  = connectivityRow[7];
            if ((left1 == -1) && (right1 == -1)) {
                throw std::runtime_error("Should not have walls to left and right of same cell");
            }

            // TODO: I don't think this is actually valid for cells that are more than 1 cell away from the boundary?
            if (left1 == -1) {
                const auto ind = right0*numDofPerCell;
                for (int i = 0; i < numDofPerCell; ++i) {
                    ghostValues[numDofPerCell + i] = currentState(ind+i);
                }
                ghostValues[numDofPerCell + 1] *= -1.0; // reverse x-momentum
            }
            if (right1 == -1) {
                const auto ind = left0*numDofPerCell;
                for (int i = 0; i < numDofPerCell; ++i) {
                    ghostValues[numDofPerCell + i] = currentState(ind+i);
                }
                ghostValues[numDofPerCell + 1] *= -1.0; // reverse x-momentum
            }
        }
    }

    template<class ConnecRowType, class FactorsType>
    void SlipWallVertBC(
        ConnecRowType const & connectivityRow,
        const double cellX, const double cellY,
        int numDofPerCell, FactorsType & factorsForBCJac) const
    {
        for (int i = 0; i < numDofPerCell; ++i) {
            factorsForBCJac[i] = 1.0;
        }
        factorsForBCJac[1] = -1.0;
    }

    template<class ConnecRowType, class StateT, class T>
    void SlipWallHorizBC(
        const int /*unused*/, ConnecRowType const & connectivityRow,
        const double cellX, const double cellY,
        const StateT & currentState, int numDofPerCell,
        const double cellWidth, T & ghostValues) const
    {
        // TODO: generalize to 3D

        // this operates under the assumption that this cell does not have ghost cells in two parallel walls
        int stencilSize1D = ghostValues.cols() / numDofPerCell;
        const int cellGID = connectivityRow[0];
        const auto uIndex  = cellGID * numDofPerCell;

        const auto front0  = connectivityRow[2];
        const auto back0  = connectivityRow[4];
        if ((front0 == -1) && (back0 == -1)) {
            throw std::runtime_error("Should not have walls to left and right of same cell");
        }

        if ((front0 == -1) || (back0 == -1)) {
            for (int i = 0; i < numDofPerCell; ++i) {
                ghostValues[i] = currentState(uIndex+i);
            }
            ghostValues[2] *= -1.0; // reverse y-momentum
        }

        // TODO: extend to WENO5
        if (stencilSize1D > 1) {
            const auto front1  = connectivityRow[6];
            const auto back1  = connectivityRow[8];
            if ((front1 == -1) && (back1 == -1)) {
                throw std::runtime_error("Should not have walls to left and right of same cell");
            }

            if (front1 == -1) {
                const auto ind = back0*numDofPerCell;
                for (int i = 0; i < numDofPerCell; ++i) {
                    ghostValues[numDofPerCell + i] = currentState(ind+i);
                }
                ghostValues[numDofPerCell + 2] *= -1.0; // reverse y-momentum
            }
            if (back1 == -1) {
                const auto ind = front0*numDofPerCell;
                for (int i = 0; i < numDofPerCell; ++i) {
                    ghostValues[numDofPerCell + i] = currentState(ind+i);
                }
                ghostValues[numDofPerCell + 2] *= -1.0; // reverse y-momentum
            }
        }
    }

    template<class ConnecRowType, class FactorsType>
    void SlipWallHorizBC(
        ConnecRowType const & connectivityRow,
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
    void SchwarzDirichletBC(
        const int gRow, ConnecRowType const & connectivityRow,
        const double cellX, const double cellY,
        const StateT & currentState, int numDofPerCell,
        const double cellWidth, T & ghostValues) const
    {
        if ((m_stateBcs == nullptr) || (m_graphBcs == nullptr)) {
            std::runtime_error("m_stateBcs or m_graphBcs not set");
        }

        // gRow: the index of current cell within graphRowsOfCellsNearBd()
        // connectivityRow: the stencil mesh graph associated with the current cell
        // ghostValues: the row of m_ghost(Left/Right/etc) associated with this cell

        int stencilSize1D = ghostValues.cols() / numDofPerCell;
        for (int stencilIdx = 0; stencilIdx < stencilSize1D; ++stencilIdx) {
            auto bcIdx = (*m_graphBcs)(gRow, stencilIdx);
            if (bcIdx != -1) {
                for (int dofIdx = 0; dofIdx < numDofPerCell; ++dofIdx) {
                    ghostValues[stencilIdx * numDofPerCell + dofIdx] = (*m_stateBcs)(bcIdx * numDofPerCell + dofIdx);
                }
            }
        }

    }

    template<class ConnecRowType, class FactorsType>
    void SchwarzDirichletBC(
        ConnecRowType const & connectivityRow,
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
	  throw std::runtime_error("Invalid probId for getPhysBCs()");

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
	      throw std::runtime_error("Unexpected GhostRelativeLocation");
            }
            break;

        default:
	  throw std::runtime_error("Invalid probId for getPhysBCs()");

    }
}

}

#endif
