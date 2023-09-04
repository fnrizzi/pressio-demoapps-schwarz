
#ifndef PRESSIODEMOAPPS_SCHWARZ_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_HPP_

#include "pressio/ode_steppers_implicit.hpp"
#include "pressiodemoapps/impl/ghost_relative_locations.hpp"
#include "./custom_bcs.hpp"
#include "./subdomain.hpp"
#include "./tiling.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;

namespace pdaschwarz{

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;

using mesh_t = pressiodemoapps::cellcentered_uniform_mesh_eigen_type;
using euler2d_app_type =
    decltype(pda::create_problem_eigen(
            declval<mesh_t>(),
            declval<pressiodemoapps::Euler2d>(),
            declval<pda::InviscidFluxReconstruction>(),
            declval<BCFunctor<mesh_t>>(),
            declval<BCFunctor<mesh_t>>(),
            declval<BCFunctor<mesh_t>>(),
            declval<BCFunctor<mesh_t>>(),
            int() /* initial condition */
        )
    );
using swe2d_app_type =
    decltype(pda::create_problem_eigen(
            declval<mesh_t>(),
            declval<pressiodemoapps::Swe2d>(),
            declval<pda::InviscidFluxReconstruction>(),
            declval<BCFunctor<mesh_t>>(),
            declval<BCFunctor<mesh_t>>(),
            declval<BCFunctor<mesh_t>>(),
            declval<BCFunctor<mesh_t>>(),
            int(), /* dummy initial */
            std::unordered_map<std::string, typename mesh_t::scalar_type>() /* user parameters */
        )
    );


template<class mesh_t, class app_t>
class SchwarzDecomp
{

public:

    using graph_t = typename app_t::mesh_connectivity_graph_type;
    using state_t = typename app_t::state_type;

    SchwarzDecomp(vector<shared_ptr<SubdomainBase<mesh_t, app_t>>> & subdomains,
                shared_ptr<const Tiling> tiling,
                vector<double> & dtVec)
        : m_tiling(tiling)
        , m_subdomainVec(subdomains)
    {
        m_dofPerCell = m_subdomainVec[0]->m_app->numDofPerCell();

        setup_controller(dtVec);
        for (int domIdx = 0; domIdx < (int) m_subdomainVec.size(); ++domIdx) {
            m_subdomainVec[domIdx]->allocateStorageForHistory(m_controlItersVec[domIdx]);
        }

        // set up communication patterns
        // FIXME: need to move this m_bcStencilSize somwwhere else
        m_bcStencilSize = 1; ///(pressiodemoapps::reconstructionTypeToStencilSize(order) - 1) / 2;
        // check_mesh_compat(); // a little error checking
        calc_exch_graph(m_bcStencilSize);

        // first communication
        for (int domIdx = 0; domIdx < (int) m_subdomainVec.size(); ++domIdx) {
            broadcast_bcState(domIdx);
        }

        // set up ghost filling graph, boundary pointers
        calc_ghost_graph();
        assignBCPointers();

    }

private:

    void setup_controller(vector<double> & dtVec)
    {
        const auto & tiling = *m_tiling;

        // physical time step checks
        m_dt = dtVec;
        if (m_dt.size() == 1) {
            m_dt.resize(tiling.count(), m_dt[0]);
        } else {
            if (m_dt.size() != (size_t) tiling.count()) {
                cerr << "m_dt.size() must be 1 or ndomains, exiting" << endl;
                exit(-1);
            }
        }
        m_dtMax = *max_element(m_dt.begin(), m_dt.end());

        // controller time step checks
        m_controlItersVec.resize(tiling.count());
        for (int domIdx = 0; domIdx < (int) m_dt.size(); ++domIdx) {
            double niters = m_dtMax / m_dt[domIdx];
            if (round(niters) == niters) {
                m_controlItersVec[domIdx] = int(round(niters));
            } else {
                cerr << "dt of domain " << domIdx
                << " (" << m_dt[domIdx]
                << ") is not an integer divisor of maximum m_dt ("
                << m_dtMax << ")" << endl;
                exit(-1);
            }
        }
    }

    void check_mesh_compat()
    {
        const auto & tiling = *m_tiling;
        const auto exchDomIds = tiling.exchDomIdVec();

        // TODO: extend this for differing (but aligned) mesh resolutions
        if (tiling.dim() == 1) return; // TODO: still need to check for differing 1D resolutions

        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

            int nx = m_subdomainVec[domIdx]->nx();
            int ny = m_subdomainVec[domIdx]->ny();
            int nz = m_subdomainVec[domIdx]->nz();

            for (int neighIdx = 0; neighIdx < (int) exchDomIds[domIdx].size(); ++neighIdx) {

                int neighDomIdx = exchDomIds[domIdx][neighIdx];
                if (neighDomIdx == -1) {
                    continue;  // not a Schwarz BC
                }

                int nxNeigh = m_subdomainVec[neighDomIdx]->nx();
                int nyNeigh = m_subdomainVec[neighDomIdx]->ny();
                int nzNeigh = m_subdomainVec[neighDomIdx]->nz();

                string xerr = "Mesh x-dimension mismatch for domains " + to_string(domIdx) + " v " + to_string(neighDomIdx) + ": " + to_string(nx) + " != " + to_string(nxNeigh);
                string yerr = "Mesh y-dimension mismatch for domains " + to_string(domIdx) + " v " + to_string(neighDomIdx) + ": " + to_string(ny) + " != " + to_string(nyNeigh);
                string zerr = "Mesh z-dimension mismatch for domains " + to_string(domIdx) + " v " + to_string(neighDomIdx) + ": " + to_string(nz) + " != " + to_string(nzNeigh);

                // left and right
                if ((neighIdx == 0) || (neighIdx == 2)) {
                    if (ny != nyNeigh) {
                        cerr << yerr << endl;
                        exit(-1);
                    }
                    if (nz != nzNeigh) {
                        cerr << zerr << endl;
                        exit(-1);
                    }
                }

                // front and back
                if ((neighIdx == 1) || (neighIdx == 3)) {
                    if (nx != nxNeigh) {
                        cerr << xerr << endl;
                        exit(-1);
                    }
                    if (nz != nzNeigh) {
                        cerr << zerr << endl;
                        exit(-1);
                    }
                }

                // bottom and top
                if ((neighIdx == 4) || (neighIdx == 5)) {
                    if (nx != nxNeigh) {
                        cerr << xerr << endl;
                        exit(-1);
                    }
                    if (ny != nyNeigh) {
                        cerr << yerr << endl;
                        exit(-1);
                    }
                }

            } // domain loop
        } // neightbor loop
    }

    void calc_exch_graph(const int bcStencil)
    {
        // TODO: extend to 3D

        // BC cell indexing example
        // L/R is from bottom to top, F/B is from left to right
        // Trying to mix cell ordering and face ordering
        //                  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        //                 ¦_(4, 1)_¦_(5, 1)_¦_(6, 1)_¦_(7, 1)_¦_(8, 1)_¦
        //  _ _ _ _ _ _ _ _¦_(4, 0)_¦_(5, 0)_¦_(6, 0)_¦_(7, 0)_¦_(8, 0)_¦_ _ _ _ _ _ _ _ _
        // ¦_(3,1)_¦_(3,0)_|________|________|________|________|________|_(12,1)_¦_(12,0)_|
        // ¦_(2,1)_¦_(2,0)_|________|________|________|________|________|_(11,1)_¦_(11,0)_|
        // ¦_(1,1)_¦_(1,0)_|________|________|________|________|________|_(10,1)_¦_(10,0)_|
        // ¦_(0,1)_¦_(0,0)_|________|________|________|________|________|_(9, 1)_¦_(9, 0)_|
        //                 ¦_(13,0)_¦_(14,0)_¦_(15,0)_¦_(16,0)_¦_(17,0)_¦
        //                 ¦_(13,1)_¦_(14,1)_¦_(15,1)_¦_(16,1)_¦_(17,1)_¦

        const auto & tiling = *m_tiling;
        m_exchGraphVec.resize(tiling.count());

        const auto overlap = tiling.overlap();
        const auto exchDomIds = tiling.exchDomIdVec();
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

            // this domain's mesh and dimensions
            int nx = m_subdomainVec[domIdx]->nx();
            int ny = m_subdomainVec[domIdx]->ny();

            // TODO: generalize to 3D
            pda::resize(m_exchGraphVec[domIdx], 2*nx + 2*ny, bcStencil);
            m_exchGraphVec[domIdx].fill(-1);

            // loop through neighboring domains
            for (int neighIdx = 0; neighIdx < (int) exchDomIds[domIdx].size(); ++neighIdx) {

                int neighDomIdx = exchDomIds[domIdx][neighIdx];
                if (neighDomIdx == -1) {
                    continue;  // not a Schwarz BC
                }

                // neighboring domain mesh and dimensions
                int nxNeigh = m_subdomainVec[neighDomIdx]->nx();
                int nyNeigh = m_subdomainVec[neighDomIdx]->ny();

                int exchCellIdx;

                // east-west neighbors will share a row index
                // left
                if (neighIdx == 0) {
                    int bcCellIdx = 0; // left boundary is the start
                    for (int yIdx = 0; yIdx < ny; ++yIdx) {
                        for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                        exchCellIdx = (nxNeigh * (yIdx + 1) - 1) - overlap - stencilIdx;
                        m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                        }
                        bcCellIdx++;
                    }
                }

                // north-south neighbors will share a column index
                // "front"
                if (neighIdx == 1) {
                    int bcCellIdx = ny * bcStencil;  // skip left boundary indices
                    for (int xIdx = 0; xIdx < nx; ++xIdx) {
                        for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                            exchCellIdx = (overlap + stencilIdx) * nxNeigh + xIdx;
                            m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                        }
                        bcCellIdx++;
                    }
                }

                // right
                if (neighIdx == 2) {
                    int bcCellIdx = (ny + nx) * bcStencil; // skip left and "front" boundary indices
                    for (int yIdx = 0; yIdx < ny; ++yIdx) {
                        for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                            exchCellIdx = nxNeigh * yIdx + overlap + stencilIdx;
                            m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                        }
                        bcCellIdx++;
                    }
                }

                // "back"
                if (neighIdx == 3) {
                    int bcCellIdx = (2*ny + nx) * bcStencil;  // skip left, "front", and right boundary indices
                    for (int xIdx = 0; xIdx < nx; ++xIdx) {
                        for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                            exchCellIdx = (nyNeigh - 1 - overlap - stencilIdx) * nxNeigh + xIdx;
                            m_exchGraphVec[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                        }
                        bcCellIdx++;
                    }
                }

                // TODO: generalize to 3D

            } // neighbor loop
        } // domain loop
    }

    template<class state_t>
    void comm_stateBc(
        const int startIdx,
        const int endIdx,
        const graph_t & exchGraph,
        state_t & bcState,
        const state_t & intState)
    {
        int exchCellIdx;

        for (int bcCellIdx = startIdx; bcCellIdx < endIdx; ++bcCellIdx) {
            for (int stencilIdx = 0; stencilIdx < m_bcStencilSize; ++stencilIdx) {
                exchCellIdx = exchGraph(bcCellIdx, stencilIdx);
                for (int dof = 0; dof < m_dofPerCell; ++dof) {
                    bcState((bcCellIdx + stencilIdx) * m_dofPerCell + dof) = intState(exchCellIdx * m_dofPerCell + dof);
                }
            }
        }
    }

    void broadcast_bcState(const int domIdx)
    {
        const auto & tiling = *m_tiling;
        const auto & exchDomIdVec = tiling.exchDomIdVec();
        const auto* domState = &m_subdomainVec[domIdx]->m_state;

        int startIdx, endIdx;
        for (int neighIdx = 0; neighIdx < (int) exchDomIdVec[domIdx].size(); ++neighIdx) {

            int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
            if (neighDomIdx == -1) {
                continue;  // not a Schwarz BC
            }

            int nxNeigh = m_subdomainVec[neighDomIdx]->nx();
            int nyNeigh = m_subdomainVec[neighDomIdx]->ny();
            auto* neighStateBCs = &m_subdomainVec[neighDomIdx]->m_stateBCs;
            const auto & neighExchGraph = m_exchGraphVec[neighDomIdx];

            // TODO: extend to 3D, need to change L/R and F/B indices to account for nzNeigh

            // this domain is the neighboring domain's left neighbor
            if (neighIdx == 2) {
                startIdx = 0;
                endIdx = nyNeigh;
                comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
            }

            // this domain is the neighboring domain's front neighbor
            if (neighIdx == 3) {
                startIdx = nyNeigh;
                endIdx = nyNeigh + nxNeigh;
                comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
            }

            // this domain is the neighboring domain's right neighbor
            if (neighIdx == 0) {
                startIdx = nyNeigh + nxNeigh;
                endIdx = 2 * nyNeigh + nxNeigh;
                comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
            }

            // this domain is the neighboring domain's back neighbor
            if (neighIdx == 1) {
                startIdx = 2 * nyNeigh + nxNeigh;
                endIdx = 2 * nyNeigh + 2 * nxNeigh;
                comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBCs, *domState);
            }

            // this domain is the neighboring domain's bottom neighbor
            // if (neighIdx == 5) {
            //   startIdx = ;
            //   endIdx = ;
            // }

            // this domain is the neighboring domain's top neighbor
            // if (neighIdx == 4) {
            //   startIdx = ;
            //   endIdx = ;
            // }

            // comm_stateBc(startIdx, endIdx, neighExchGraph, *neighStateBc, domState);

        }
    }

    void calc_ghost_graph()
    {
        const auto & tiling = *m_tiling;
        const auto & exchDomIdVec = tiling.exchDomIdVec();

        m_ghostGraphVec.resize(tiling.count());
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

            const auto & meshObj = *(m_subdomainVec[domIdx]->m_mesh);
            const auto intGraph = meshObj.graph();
            int nx = m_subdomainVec[domIdx]->nx();
            int ny = m_subdomainVec[domIdx]->ny();
            // int nz = m_subdomainVec[domIdx]->nz();

            // unique mask for each subdomain edge
            const auto & rowsBd = meshObj.graphRowsOfCellsNearBd();
            m_ghostGraphVec[domIdx].resize(2 * tiling.dim());
            for (int bcIdx = 0; bcIdx < 2 * tiling.dim(); ++bcIdx) {
                m_ghostGraphVec[domIdx][bcIdx].resize(int(rowsBd.size()), -1);
            }

            for (decltype(rowsBd.size()) it = 0; it < rowsBd.size(); ++it) {

                // ASK FR: is there an instance when rowsBd[it] != intGraph(rowsBd[it], 0)?
                //    The indices appear to be identical
                // TODO: this is all totally wrong for higher order

                const auto smPt = rowsBd[it];
                const auto left0  = intGraph(smPt, 1);
                const auto front0 = intGraph(smPt, 2);
                const auto right0 = intGraph(smPt, 3);
                const auto back0  = intGraph(smPt, 4);

                // int stencilIdx = 0; // first order
                int rowIdx = smPt / nx;
                int colIdx = smPt % nx;
                int bcCellIdx;

                if (left0 == -1) {
                    if (exchDomIdVec[domIdx][0] != -1) {
                        bcCellIdx = rowIdx;
                        m_ghostGraphVec[domIdx][0][it] = bcCellIdx * m_dofPerCell;
                    }
                }

                if (front0 == -1) {
                    if (exchDomIdVec[domIdx][1] != -1) {
                        bcCellIdx = ny + colIdx;
                        m_ghostGraphVec[domIdx][1][it] = bcCellIdx * m_dofPerCell;
                    }
                }

                if (right0 == -1) {
                    if (exchDomIdVec[domIdx][2] != -1) {
                        bcCellIdx = ny + nx + rowIdx;
                        m_ghostGraphVec[domIdx][2][it] = bcCellIdx * m_dofPerCell;
                    }
                }

                if (back0 == -1) {
                    if (exchDomIdVec[domIdx][3] != -1) {
                        bcCellIdx = 2 * ny + nx + colIdx;
                        m_ghostGraphVec[domIdx][3][it] = bcCellIdx * m_dofPerCell;
                    }
                }
                // TODO: extend to higher order, 3D

            } // boundary cell loop
        } // domain loop
    }

    // assign pointers in BCFunctors
    void assignBCPointers()
    {

        const auto & tiling = *m_tiling;
        const auto & exchDomIdVec = tiling.exchDomIdVec();

        for (int domIdx = 0; domIdx < (int) m_subdomainVec.size(); ++domIdx) {
            for (int neighIdx = 0; neighIdx < (int) exchDomIdVec[domIdx].size(); ++neighIdx) {

                int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
                    if (neighDomIdx == -1) {
                    continue;  // not a Schwarz BC
                }

                // has left neighbor
                if (neighIdx == 0) {
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Left, &m_subdomainVec[domIdx]->m_stateBCs);
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Left, &m_ghostGraphVec[domIdx][0]);
                }

                // has front neighbor
                if (neighIdx == 1) {
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Front, &m_subdomainVec[domIdx]->m_stateBCs);
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Front, &m_ghostGraphVec[domIdx][1]);
                }

                // has right neighbor
                if (neighIdx == 2) {
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Right, &m_subdomainVec[domIdx]->m_stateBCs);
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Right, &m_ghostGraphVec[domIdx][2]);
                }

                // has back neighbor
                if (neighIdx == 3) {
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Back, &m_subdomainVec[domIdx]->m_stateBCs);
                    m_subdomainVec[domIdx]->m_app->setBCPointer(pda::impl::GhostRelativeLocation::Back, &m_ghostGraphVec[domIdx][3]);
                }
            }
        }
    }

    template <class state_t>
    array<double, 2> calcConvergence(const state_t & state1, const state_t & state2)
    {
        // TODO: assumed to be an Eigen state, not sure how to generalize
        // TODO: compute convergence for each variable separately

        int numDOF = state1.size();
        if (state2.size() != numDOF) {
            cerr << "state1 size does not match state2 size, " << numDOF << " vs. " << state2.size() << endl;
            exit(-1);
        }

        // absolute error
        double abs_err = (state1 - state2).squaredNorm();

        // handle edge cases for relative error
        double rel_err;
        double basenorm = state1.squaredNorm();
        if (basenorm > 0) {
            rel_err = abs_err / basenorm;
        }
        else {
            if (abs_err > 0) {
                rel_err = 1.0;
            }
            else {
                rel_err = 0.0;
            }
        }

        array<double, 2> errArr = {abs_err, rel_err};
        return errArr;

    }

public:

    void calc_controller_step(
        int outerStep,
        double time,
        const double rel_err_tol,
        const double abs_err_tol,
        const int convergeStepMax,
        const bool additive)
    {

        const auto & tiling = *m_tiling;
        const auto ndomains = tiling.count();

        // store initial step for resetting if Schwarz iter does not converge
        for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
            m_subdomainVec[domIdx]->storeStateHistory(0);
        }

        // convergence
        int convergeStep = 0;
        vector<array<double, 2>> convergeVals(ndomains);
        while (convergeStep < convergeStepMax) {

            cout << "Schwarz iteration " << convergeStep + 1 << endl;

            for (int domIdx = 0; domIdx < ndomains; ++domIdx) {

                // reset to beginning of controller time
                auto timeDom = time;
                auto stepDom = outerStep * m_controlItersVec[domIdx];

                const auto dtDom = m_dt[domIdx];
                const auto dtWrap = pode::StepSize<double>(dtDom);

                // controller inner loop
                for (int innerStep = 0; innerStep < m_controlItersVec[domIdx]; ++innerStep) {

                    const auto startTimeWrap = pode::StepStartAt<double>(timeDom);
                    const auto stepWrap = pode::StepCount(stepDom);

                    m_subdomainVec[domIdx]->doStep(startTimeWrap, stepWrap, dtWrap);
                    m_subdomainVec[domIdx]->updateFullState(); // noop for FOM subdomain

                    // for last iteration, compute convergence criteria
                    // important to do this before saving history, as stateHistVec still has last convergence loop's state
                    // NOTE: this is always computed on the full-order state
                    if (innerStep == (m_controlItersVec[domIdx] - 1)) {
                        convergeVals[domIdx] = calcConvergence(m_subdomainVec[domIdx]->m_state, m_subdomainVec[domIdx]->m_stateHistVec.back());
                    }

                    // store intra-step history
                    m_subdomainVec[domIdx]->storeStateHistory(innerStep+1);

                    // set (interpolated) boundary conditions

                    // update local step and time
                    stepDom++;
                    timeDom += dtDom;

                } // domain loop

                // broadcast boundary conditions immediately for multiplicative Schwarz
                if (!additive) {
                    broadcast_bcState(domIdx);
                }

            }

            // check convergence for all domains, break if conditions met
            double abs_err = 0.0;
            double rel_err = 0.0;
            for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
                abs_err += convergeVals[domIdx][0];
                rel_err += convergeVals[domIdx][1];
            }
            abs_err /= ndomains;
            rel_err /= ndomains;
            cout << "Average abs err: " << abs_err << endl;
            cout << "Average rel err: " << rel_err << endl;
            if ((rel_err < rel_err_tol) || (abs_err < abs_err_tol)) {
                break;
            }

            // broadcast boundary conditions after domain cycle for additive Schwarz
            if (additive) {
                for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
                    broadcast_bcState(domIdx);
                }
            }

            convergeStep++;

            // reset interior state if not converged
            for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
                m_subdomainVec[domIdx]->resetStateFromHistory();
            }

        } // convergence loop
    }


public:

    int m_dofPerCell;
    shared_ptr<const Tiling> m_tiling;
    vector<shared_ptr<SubdomainBase<mesh_t, app_t>>> & m_subdomainVec;
    int m_bcStencilSize;
    double m_dtMax;
    vector<double> m_dt;
    vector<graph_t> m_exchGraphVec;
    vector<vector<vector<int>>> m_ghostGraphVec; // 1: subdomain index, 2: edge index, 3: cell index
    vector<int> m_controlItersVec;

};
}

#endif