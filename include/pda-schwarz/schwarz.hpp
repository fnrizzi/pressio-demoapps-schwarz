
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
#include <chrono>


namespace pdaschwarz{

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;

using mesh_t = pressiodemoapps::cellcentered_uniform_mesh_eigen_type;
using euler2d_app_type =
    decltype(pda::create_problem_eigen(
            std::declval<mesh_t>(),
            std::declval<pressiodemoapps::Euler2d>(),
            std::declval<pda::InviscidFluxReconstruction>(),
            std::declval<BCFunctor<mesh_t>>(),
            std::declval<BCFunctor<mesh_t>>(),
            std::declval<BCFunctor<mesh_t>>(),
            std::declval<BCFunctor<mesh_t>>(),
            int(), /* initial condition */
            std::unordered_map<std::string, typename mesh_t::scalar_type>() /* user parameters */
        )
    );
using swe2d_app_type =
    decltype(pda::create_problem_eigen(
            std::declval<mesh_t>(),
            std::declval<pressiodemoapps::Swe2d>(),
            std::declval<pda::InviscidFluxReconstruction>(),
            std::declval<BCFunctor<mesh_t>>(),
            std::declval<BCFunctor<mesh_t>>(),
            std::declval<BCFunctor<mesh_t>>(),
            std::declval<BCFunctor<mesh_t>>(),
            int(), /* dummy initial */
            std::unordered_map<std::string, typename mesh_t::scalar_type>() /* user parameters */
        )
    );


template<class ...SubdomainArgs>
class SchwarzDecomp
{

public:
    using subdomain_base_t = SubdomainBase<SubdomainArgs...>;
    using graph_t = typename subdomain_base_t::mesh_t::graph_t;
    using state_t = typename subdomain_base_t::state_t;

    SchwarzDecomp(std::vector<std::shared_ptr< subdomain_base_t >> & subdomains,
                std::shared_ptr<const Tiling> tiling,
                std::vector<double> & dtVec)
        : m_tiling(tiling)
        , m_subdomainVec(subdomains)
    {
        m_dofPerCell = m_subdomainVec[0]->getDofPerCell();

        setup_controller(dtVec);
        for (int domIdx = 0; domIdx < (int) m_subdomainVec.size(); ++domIdx) {
            m_subdomainVec[domIdx]->allocateStorageForHistory(m_controlItersVec[domIdx]);
        }

        // set up communication patterns, first communication
        calc_exch_graph();
        for (int domIdx = 0; domIdx < (int) m_subdomainVec.size(); ++domIdx) {
            broadcast_bcState(domIdx);
        }

        // set up ghost filling graph, boundary pointers
        calc_ghost_graph();

    }

private:

    void setup_controller(std::vector<double> & dtVec)
    {
        const auto & tiling = *m_tiling;

        // physical time step checks
        m_dt = dtVec;
        if (m_dt.size() == 1) {
            m_dt.resize(tiling.count(), m_dt[0]);
        } else {
            if (m_dt.size() != (size_t) tiling.count()) {
                std::cerr << "m_dt.size() must be 1 or ndomains, exiting" << std::endl;
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
                std::cerr << "dt of domain " << domIdx
                << " (" << m_dt[domIdx]
                << ") is not an integer divisor of maximum m_dt ("
                << m_dtMax << ")" << std::endl;
                exit(-1);
            }
        }
    }

    // void build_neighbor_connectivity()
    // {
    //     const auto & tiling = *m_tiling;
        
    //     std::vector<std::vector<std::vector<std::vector<int>>>> global_to_stencil_map_sub;
        
    //     global_to_stencil_map_sub.resize(tiling.countX());
    //     for (int i = 0; i < tiling.countX(); ++i) {
    //         global_to_stencil_map_sub[i].resize(tiling.countY());
    //         for (int j = 0; j < tiling.countY(); ++j) {
    //             global_to_stencil_map_sub[i][j].resize(tiling.countZ());
    //             for (int k = 0; k < tiling.countZ(); ++k) {
    //                 int dom_idx = i + j * tiling.countX() + k * tiling.countX() * tiling.countY();
    //                 const auto & meshFullObj = m_subdomainVec[dom_idx]->getFullMesh();
    //                 global_to_stencil_map_sub[i][j][k].resize();
                    
    //             }
    //         }
    //     }

    // }

    // determines whether LOCAL neighbor orientation indices correspond to the same neighbors
    // deals with weird ordering difference in 1D
    bool is_neighbor_pair(int id1, int id2)
    {
        if (id1 == id2) {
            return false;
        }
        // increasing order
        if (id1 > id2) {
            int temp = id2;
            id2 = id1;
            id1 = temp;
        }

        if ((*m_tiling).dim() == 1) {
            if ((id1 == 0) && (id2 == 1)) { return true; }
            else { return false; }
        }
        else if ((*m_tiling).dim() >= 2) {
            if ((id1 == 0) && (id2 == 2)) { return true; }
            else if ((id1 == 1) && (id2 == 3)) { return true; }

            if ((*m_tiling).dim() == 3) {
                if ((id1 == 4) && (id2 == 5)) { return true; }
                else { return false; }
            }
            else {
                return false;
            }
        }

        return false;
    }

    void calc_exch_graph()
    {
        // TODO: extend to 3D

        const auto & tiling = *m_tiling;
        m_broadcastGraphVec.resize(tiling.count());

        const auto exchDomIds = tiling.exchDomIdVec();
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

            // entry for every possible neighbor
            m_broadcastGraphVec[domIdx].resize(2 * tiling.dim());

            // determine broadcast pattern to neighbors
            for (int neighIdx = 0; neighIdx < (int) exchDomIds[domIdx].size(); ++neighIdx) {

                int neighDomIdx = exchDomIds[domIdx][neighIdx];
                if (neighDomIdx == -1) {
                    continue;  // not a Schwarz BC
                }

                const auto & neighMeshObj = m_subdomainVec[neighDomIdx]->getMeshStencil();
                const auto & neighNeighborGraph = m_subdomainVec[neighDomIdx]->getNeighborGraph();
                const auto & neighRowsBd = neighMeshObj.graphRowsOfCellsNearBd();

                // count number of cells to be broadcast to this neighbor
                // TODO: for true parallelism, can just split this as the send/recv indices
                int broadcastCount = 0;
                int ghostCount = 0;
                for (int bdIdx = 0; bdIdx < neighRowsBd.size(); ++bdIdx) {
                    auto rowIdx = neighRowsBd[bdIdx];
                    for (int colIdx = 1; colIdx < neighNeighborGraph.cols(); ++colIdx) {
                        int broadcastGID = neighNeighborGraph(rowIdx, colIdx);
                        if (broadcastGID != -1) {
                            int gatherIdx = (colIdx - 1) % (2 * tiling.dim());
                            if (is_neighbor_pair(neighIdx, gatherIdx)) {
                                m_broadcastGraphVec[domIdx][neighIdx].push_back({broadcastGID, ghostCount});
                            }
                            ghostCount++;
                        }
                    }
                }

            } // neighbor loop
        } // domain loop
    }

    void broadcast_bcState(const int domIdx)
    {
        const auto & tiling = *m_tiling;
        const auto & exchDomIdVec = tiling.exchDomIdVec();
        const auto * state = m_subdomainVec[domIdx]->getStateStencil();

        for (auto neighIdx = 0; neighIdx < exchDomIdVec[domIdx].size(); ++neighIdx) {

            int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
            if (neighDomIdx == -1) {
                continue;  // not a Schwarz BC
            }

            auto * neighStateBCs = m_subdomainVec[neighDomIdx]->getStateBCs();

            for (auto bcIdx = 0; bcIdx < m_broadcastGraphVec[domIdx][neighIdx].size(); ++bcIdx) {
                auto sourceGID = m_broadcastGraphVec[domIdx][neighIdx][bcIdx][0];
                auto targetBCID = m_broadcastGraphVec[domIdx][neighIdx][bcIdx][1];
                for (auto dofIdx = 0; dofIdx < m_dofPerCell; ++dofIdx) {
                    (*neighStateBCs)(targetBCID * m_dofPerCell + dofIdx) = (*state)(sourceGID * m_dofPerCell + dofIdx);
                }
            }
        }
    }

    void calc_ghost_graph()
    {
        const auto & tiling = *m_tiling;
        const auto & exchDomIdVec = tiling.exchDomIdVec();

        m_ghostGraphVec.resize(tiling.count());
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {

            const auto & meshObj = m_subdomainVec[domIdx]->getMeshStencil();
            const auto & neighborGraph = m_subdomainVec[domIdx]->getNeighborGraph();
            const auto & rowsBd = meshObj.graphRowsOfCellsNearBd();
            const auto stencilSize1D = (meshObj.stencilSize() - 1) / 2;

            m_ghostGraphVec[domIdx].resize(2 * tiling.dim());
            for (int neighIdx = 0; neighIdx < exchDomIdVec[domIdx].size(); ++neighIdx) {
                pda::resize(m_ghostGraphVec[domIdx][neighIdx], (int) rowsBd.size(), stencilSize1D);
                m_ghostGraphVec[domIdx][neighIdx].fill(-1);
            }

            int ghostCount = 0;
            for (auto bdIdx = 0; bdIdx < rowsBd.size(); ++bdIdx) {
                int rowIdx = rowsBd[bdIdx];
                for (int colIdx = 1; colIdx < neighborGraph.cols(); ++colIdx) {
                    int nieghGID = neighborGraph(rowIdx, colIdx);
                    if (nieghGID != -1) {
                        int neighIdx = (colIdx - 1) % (2 * tiling.dim());
                        int stencilIdx = (colIdx - 1) / (2 * tiling.dim());

                        m_ghostGraphVec[domIdx][neighIdx](bdIdx, stencilIdx) = ghostCount;

                        ghostCount++;
                    }
                }
            } // boundary cell loop

            for (int neighIdx = 0; neighIdx < exchDomIdVec[domIdx].size(); ++neighIdx) {
                int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
                    if (neighDomIdx == -1) {
                    continue;  // not a Schwarz BC
                }

                // left neighbor
                if (neighIdx == 0) {
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Left, m_subdomainVec[domIdx]->getStateBCs());
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Left, &m_ghostGraphVec[domIdx][0]);
                }

                // front neighbor
                if (neighIdx == 1) {
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Front, m_subdomainVec[domIdx]->getStateBCs());
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Front, &m_ghostGraphVec[domIdx][1]);
                }

                // right neighbor
                if (neighIdx == 2) {
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Right, m_subdomainVec[domIdx]->getStateBCs());
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Right, &m_ghostGraphVec[domIdx][2]);
                }

                // back neighbor
                if (neighIdx == 3) {
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Back, m_subdomainVec[domIdx]->getStateBCs());
                    m_subdomainVec[domIdx]->setBCPointer(pda::impl::GhostRelativeLocation::Back, &m_ghostGraphVec[domIdx][3]);
                }

                // TODO: expand to 3D

            } // neighbor loop
        } // domain loop
    }

    template <class state_t>
    std::array<double, 2> calcConvergence(const state_t & state1, const state_t & state2)
    {
        // TODO: assumed to be an Eigen state, not sure how to generalize
        // TODO: compute convergence for each variable separately

        int numDOF = state1.size();
        if (state2.size() != numDOF) {
            std::cerr << "state1 size does not match state2 size, " << numDOF << " vs. " << state2.size() << std::endl;
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

        std::array<double, 2> errArr = {abs_err, rel_err};
        return errArr;

    }

public:

    std::vector<std::vector<double>> calc_controller_step(
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
        std::vector<std::array<double, 2>> convergeVals(ndomains);
        std::vector<std::vector<double>> iterTime(ndomains);
        while (convergeStep < convergeStepMax) {

            std::cout << "Schwarz iteration " << convergeStep + 1 << std::endl;

            for (int domIdx = 0; domIdx < ndomains; ++domIdx) {

                // reset to beginning of controller time
                auto timeDom = time;
                auto stepDom = outerStep * m_controlItersVec[domIdx];

                const auto dtDom = m_dt[domIdx];
                const auto dtWrap = pode::StepSize<double>(dtDom);

                auto runtimeStart = std::chrono::high_resolution_clock::now();

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
                        convergeVals[domIdx] = calcConvergence(*m_subdomainVec[domIdx]->getStateStencil(),
                            m_subdomainVec[domIdx]->getLastStateInHistory()/*m_stateHistVec.back()*/);
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

                // record iteration runtime (in seconds)
                auto runtimeEnd = std::chrono::high_resolution_clock::now();
                double nsElapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(runtimeEnd - runtimeStart).count());
                iterTime[domIdx].emplace_back(nsElapsed * 1e-9);

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
            std::cout << "Average abs err: " << abs_err << std::endl;
            std::cout << "Average rel err: " << rel_err << std::endl;
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

        return iterTime;

    }


public:

    int m_dofPerCell;
    std::shared_ptr<const Tiling> m_tiling;
    std::vector<std::shared_ptr<subdomain_base_t>> & m_subdomainVec;
    double m_dtMax;
    std::vector<double> m_dt;
    std::vector<std::vector<std::vector<std::array<int, 2>>>> m_broadcastGraphVec;
    std::vector<std::vector<graph_t>> m_ghostGraphVec;
    std::vector<int> m_controlItersVec;

};
}

#endif
