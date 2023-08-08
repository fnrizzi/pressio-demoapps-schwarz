

#ifndef PRESSIODEMOAPPS_SCHWARZ_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "pressio/ode_steppers_implicit.hpp"

#include "pressiodemoapps/impl/ghost_relative_locations.hpp"
#include "./custom_bcs.hpp"
#include "./subdomain.hpp"

using namespace std;

namespace pdaschwarz{

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;


template<class prob_t, class mesh_t, class order_t, class scheme_t, class subdom_t>
class SchwarzDecomp
{

    using graph_t = typename mesh_t::graph_t;

    public:

        SchwarzDecomp(
            prob_t probId,
            order_t order,
            scheme_t scheme,
            const string & meshRoot,
            vector<double> & dtVec,
            const int icflag = 1)
        {

            // get decomposition info
            read_domain_info(meshRoot);
            m_ndomains = m_ndomX * m_ndomY * m_ndomZ;

            // set up problem
            setup_controller(dtVec);
            init_subdomains(probId, order, scheme, meshRoot, icflag);

            // set up communication patterns
            bcStencilSize = (pda::reconstructionTypeToStencilSize(order) - 1) / 2;
            exchDomIdVec = calc_neighbor_dims();
            check_mesh_compat(); // a little error checking
            exchGraphVec = calc_exch_graph(bcStencilSize, exchDomIdVec);

            // first communication
            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {
                broadcast_bcState(domIdx);
            }

            // set up ghost filling graph
            // ghostGraphVec = calc_ghost_graph();

        }

    private:

        void read_domain_info(const string & meshRoot)
        {
            const auto inFile = meshRoot + "/info_domain.dat";
            ifstream foundFile(inFile);
            if(!foundFile){
            cout << "file not found " << inFile << endl;
            exit(EXIT_FAILURE);
            }

            // defaults
            m_ndomX = 1;
            m_ndomY = 1;
            m_ndomZ = 1;

            ifstream source( inFile, ios_base::in);
            string line;
            while (getline(source, line) )
            {
                istringstream ss(line);
                string colVal;
                ss >> colVal;

                if (colVal == "dim"){
                    ss >> colVal;
                    m_dim = stoi(colVal);
                    if (m_dim < 1)
                        throw runtime_error("dim must be >= 1");
                }

                else if (colVal == "ndomX"){
                    ss >> colVal;
                    m_ndomX = stoi(colVal);
                    if (m_ndomX < 1)
                        throw runtime_error("ndomX must be >= 1");
                }

                else if (colVal == "ndomY"){
                    ss >> colVal;
                    m_ndomY = stoi(colVal);
                    if (m_ndomY < 1)
                        throw runtime_error("ndomY must be >= 1");
                }

                else if (colVal == "ndomZ"){
                    ss >> colVal;
                    m_ndomZ = stoi(colVal);
                    if (m_ndomZ < 1)
                        throw runtime_error("ndomZ must be >= 1");
                }

                else if (colVal == "overlap"){
                    ss >> colVal;
                    m_overlap = stoi(colVal);
                    if (m_overlap < 0)
                        throw runtime_error("overlap must be > 0");
                    // has to be an even number for simplicity, can change later
                    if (m_overlap % 2) {
                        cerr << "overlap must be an even number" << endl;
                        exit(-1);
                    }
                }
            }
            source.close();
        }

        void setup_controller(vector<double> & dtVec)
        {

            // physical time step checks
            dt = dtVec;
            if (dt.size() == 1) {
                dt.resize(m_ndomains, dt[0]);
            } else {
                if (dt.size() != (size_t) m_ndomains) {
                    cerr << "dt.size() must be 1 or ndomains, exiting" << endl;
                    exit(-1);
                }
            }
            dtMax = *max_element(dt.begin(), dt.end());

            // controller time step checks
            controlItersVec.resize(m_ndomains);
            for (int domIdx = 0; domIdx < (int) dt.size(); ++domIdx) {
                double niters = dtMax / dt[domIdx];
                if (round(niters) == niters) {
                    controlItersVec[domIdx] = int(round(niters));
                } else {
                    cerr << "dt of domain " << domIdx << " (" << dt[domIdx] << ") is not an integer divisor of maximum dt (" << dtMax << ")" << endl;
                    exit(-1);
                }
            }

        }

        void init_subdomains(
            prob_t probId,
            order_t order,
            scheme_t scheme,
            const string & meshRoot,
            const int icflag)
        {

            // TODO: generalize for 1D, 3D

            // physical boundaries
            auto bcLeftPhys  = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Left);
            auto bcRightPhys = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Right);
            auto bcFrontPhys = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Front);
            auto bcBackPhys  = getPhysBCs(probId, pda::impl::GhostRelativeLocation::Back);

            // get app type to template Subdomain, state type for later
            using app_t = decltype(
                pda::create_problem_eigen(
                    declval<mesh_t>(), probId, order,
                    BCFunctor(bcLeftPhys), BCFunctor(bcRightPhys),
                    BCFunctor(bcFrontPhys), BCFunctor(bcBackPhys),
                    icflag)
            );
            using state_t = typename app_t::state_type;

            m_dofPerCell = app_t::numDofPerCell;

            int i, j;
            BCType bcLeft, bcRight, bcFront, bcBack;

            // determine boundary conditions for each subdomain, specify app type
            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx)
            {
                i = domIdx % m_ndomX;
                j = domIdx / m_ndomX;

                // left physical boundary
                if (i == 0) {
                    auto bcLeft = bcLeftPhys;
                }
                else {
                    auto bcLeft = BCType::SchwarzDirichlet;
                }
                // right physical boundary
                if (i == (m_ndomX - 1)) {
                    auto bcRight = bcRightPhys;
                }
                else {
                    auto bcRight = BCType::SchwarzDirichlet;
                }
                // front physical boundary
                if (j == 0) {
                    auto bcFront = bcFrontPhys;
                }
                else {
                    auto bcFront = BCType::SchwarzDirichlet;
                }
                // back physical boundary
                if (j == (m_ndomY - 1)) {
                    auto bcBack = bcBackPhys;
                }
                else {
                    auto bcBack = BCType::SchwarzDirichlet;
                }

                subdomainVec.emplace_back(
                    Subdomain<prob_t, order_t, scheme_t, mesh_t, app_t>(
                        probId, order, scheme, domIdx, meshRoot,
                        bcLeft, bcFront, bcRight, bcBack,
                        controlItersVec[domIdx], icflag
                    )
                );

            }
        }

        vector<vector<int>> calc_neighbor_dims()
        {
            // determine neighboring domain IDs

            int maxDomNeighbors = 2 * m_dim;
            vector<vector<int>> exchDomIds(m_ndomains, vector<int>(maxDomNeighbors, -1));

            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {

                // subdomain indices
                int i = {};
                int j = {};
                int k = {};
                i = domIdx % m_ndomX;
                if (m_dim > 1) {
                    j = domIdx / m_ndomX;
                }
                if (m_dim == 2) {
                    k = domIdx / (m_ndomX * m_ndomY);
                }


                // 1D, 2D, and 3D
                // left boundary
                if (i != 0) {
                    exchDomIds[domIdx][0] = domIdx - 1;
                }

                // right boundary
                if (i != (m_ndomX - 1)) {
                    // ordering change for 1D vs. 2D/3D faces
                    if (m_dim == 1) {
                        exchDomIds[domIdx][1] = domIdx + 1;
                    }
                    else {
                        exchDomIds[domIdx][2] = domIdx + 1;
                    }
                }

                // 2D and 3D
                if (m_dim > 1) {
                    // front boundary
                    if (j != (m_ndomY - 1)) {
                        exchDomIds[domIdx][1] = domIdx + m_ndomX;
                    }

                    // back boundary
                    if (j != 0) {
                        exchDomIds[domIdx][3] = domIdx - m_ndomX;
                    }
                }

                // 3D
                if (m_dim > 2) {
                    // bottom boundary
                    if (k != 0) {
                        exchDomIds[domIdx][4] = domIdx - (m_ndomX * m_ndomY);
                    }

                    // top boundary
                    if (k != (m_ndomZ - 1)) {
                        exchDomIds[domIdx][5] = domIdx + (m_ndomX * m_ndomY);
                    }
                }

            }

            return exchDomIds;

        }

        void check_mesh_compat() {

            // TODO: extend this for differing (but aligned) mesh resolutions
            if (m_dim == 1) return; // TODO: still need to check for differing 1D resolutions

            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {

                int nx = subdomainVec[domIdx].nx();
                int ny = subdomainVec[domIdx].ny();
                int nz = subdomainVec[domIdx].nz();

                for (int neighIdx = 0; neighIdx < (int) exchDomIdVec[domIdx].size(); ++neighIdx) {

                    int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
                    if (neighDomIdx == -1) {
                        continue;  // not a Schwarz BC
                    }

                    int nxNeigh = subdomainVec[neighDomIdx].nx();
                    int nyNeigh = subdomainVec[neighDomIdx].ny();
                    int nzNeigh = subdomainVec[neighDomIdx].nz();

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

        vector<graph_t> calc_exch_graph(
            const int bcStencil,
            const vector<vector<int>> & exchDomIds)
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

            vector<graph_t> exchGraphs(m_ndomains);

            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {

                // this domain's mesh and dimensions
                int nx = subdomainVec[domIdx].nx();
                int ny = subdomainVec[domIdx].ny();

                // TODO: generalize to 3D
                pda::resize(exchGraphs[domIdx], 2*nx + 2*ny, bcStencil);
                exchGraphs[domIdx].fill(-1);

                // loop through neighboring domains
                for (int neighIdx = 0; neighIdx < (int) exchDomIds[domIdx].size(); ++neighIdx) {

                    int neighDomIdx = exchDomIds[domIdx][neighIdx];
                    if (neighDomIdx == -1) {
                        continue;  // not a Schwarz BC
                    }

                    // neighboring domain mesh and dimensions
                    int nxNeigh = subdomainVec[neighDomIdx].nx();
                    int nyNeigh = subdomainVec[neighDomIdx].ny();

                    int exchCellIdx;

                    // east-west neighbors will share a row index
                    // left
                    if (neighIdx == 0) {
                        int bcCellIdx = 0; // left boundary is the start
                        for (int yIdx = 0; yIdx < ny; ++yIdx) {
                            for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                            exchCellIdx = (nxNeigh * (yIdx + 1) - 1) - m_overlap - stencilIdx;
                            exchGraphs[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
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
                                exchCellIdx = (m_overlap + stencilIdx) * nxNeigh + xIdx;
                                exchGraphs[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                            }
                            bcCellIdx++;
                        }
                    }

                    // right
                    if (neighIdx == 2) {
                        int bcCellIdx = (ny + nx) * bcStencil; // skip left and "front" boundary indices
                        for (int yIdx = 0; yIdx < ny; ++yIdx) {
                            for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                                exchCellIdx = nxNeigh * yIdx + m_overlap + stencilIdx;
                                // TODO: check for a different problem
                                exchGraphs[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                            }
                            bcCellIdx++;
                        }
                    }

                    // "back"
                    if (neighIdx == 3) {
                    int bcCellIdx = (2*ny + nx) * bcStencil;  // skip left, "front", and right boundary indices
                    for (int xIdx = 0; xIdx < nx; ++xIdx) {
                        for (int stencilIdx = 0; stencilIdx < bcStencil; ++stencilIdx) {
                        exchCellIdx = (nyNeigh - 1 - m_overlap - stencilIdx) * nxNeigh + xIdx;
                        exchGraphs[domIdx](bcCellIdx, stencilIdx) = exchCellIdx;
                        }
                        bcCellIdx++;
                    }
                    }

                    // TODO: generalize to 3D

                } // neighbor loop
            } // domain loop

            return exchGraphs;

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
                for (int stencilIdx = 0; stencilIdx < bcStencilSize; ++stencilIdx) {
                    exchCellIdx = exchGraph(bcCellIdx, stencilIdx);
                    for (int dof = 0; dof < m_dofPerCell; ++dof) {
                    bcState((bcCellIdx + stencilIdx) * m_dofPerCell + dof) = intState(exchCellIdx * m_dofPerCell + dof);
                    }
                }
            }

        }

        void broadcast_bcState(const int domIdx)
        {

            const auto* domState = &subdomainVec[domIdx].state;

            int startIdx, endIdx;
            for (int neighIdx = 0; neighIdx < (int) exchDomIdVec[domIdx].size(); ++neighIdx) {

                int neighDomIdx = exchDomIdVec[domIdx][neighIdx];
                if (neighDomIdx == -1) {
                    continue;  // not a Schwarz BC
                }

                int nxNeigh = subdomainVec[neighDomIdx].nx();
                int nyNeigh = subdomainVec[neighDomIdx].ny();
                auto* neighStateBCs = &subdomainVec[neighDomIdx].stateBCs;
                const auto neighExchGraph = exchGraphVec[neighDomIdx];

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

        vector<graph_t> calc_ghost_graph()
        {

            vector<graph_t> ghostGraphs(m_ndomains);

            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {

                const auto meshObj = subdomainVec[domIdx].mesh;
                const auto intGraph = meshObj.graph();
                int nx = subdomainVec[domIdx].nx();
                int ny = subdomainVec[domIdx].ny();
                // int nz = subdomainVec[domIdx].nz();

                const auto & rowsBd = meshObj.graphRowsOfCellsNearBd();
                pda::resize(ghostGraphs[domIdx], int(rowsBd.size()), 2 * m_dim);
                ghostGraphs[domIdx].fill(-1);

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
                            ghostGraphs[domIdx](it, 0) = bcCellIdx * m_dofPerCell;
                        }
                    }

                    if (front0 == -1) {
                        if (exchDomIdVec[domIdx][1] != -1) {
                            bcCellIdx = ny + colIdx;
                            ghostGraphs[domIdx](it, 1) = bcCellIdx * m_dofPerCell;
                        }
                    }

                    if (right0 == -1) {
                        if (exchDomIdVec[domIdx][2] != -1) {
                            bcCellIdx = ny + nx + rowIdx;
                            ghostGraphs[domIdx](it, 2) = bcCellIdx * m_dofPerCell;
                        }
                    }

                    if (back0 == -1) {
                        if (exchDomIdVec[domIdx][3] != -1) {
                            bcCellIdx = 2 * ny + nx + colIdx;
                            ghostGraphs[domIdx](it, 3) = bcCellIdx * m_dofPerCell;
                        }
                    }
                    // TODO: extend to higher order, 3D

                } // boundary cell loop
            } // domain loop

            return ghostGraphs;

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
            const int convergeStepMax)
        {

            // store initial step for resetting if Schwarz iter does not converge
            for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {
                subdomainVec[domIdx].stateHistVec[0] = subdomainVec[domIdx].state;
            }

            // convergence
            int convergeStep = 0;
            vector<array<double, 2>> convergeVals(m_ndomains);
            while (convergeStep < convergeStepMax) {

                cout << "Schwarz iteration " << convergeStep + 1 << endl;

                for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {

                    // reset to beginning of controller time
                    auto timeDom = time;
                    auto stepDom = outerStep * controlItersVec[domIdx];

                    const auto dtDom = dt[domIdx];
                    const auto dtWrap = pode::StepSize<double>(dtDom);

                    // controller inner loop
                    for (int innerStep = 0; innerStep < controlItersVec[domIdx]; ++innerStep) {

                        const auto startTimeWrap = pode::StepStartAt<double>(timeDom);
                        const auto stepWrap = pode::StepCount(stepDom);

                        subdomainVec[domIdx].stepper(subdomainVec[domIdx].state, startTimeWrap, stepWrap, dtWrap, subdomainVec[domIdx].nonlinSolver);

                        // for last iteration, compute convergence criteria
                        // important to do this before saving history, as stateHistVec still has last convergence loop's state
                        if (innerStep == (controlItersVec[domIdx] - 1)) {
                            convergeVals[domIdx] = calcConvergence(subdomainVec[domIdx].state, subdomainVec[domIdx].stateHistVec.back());
                        }

                        // store intra-step history
                        subdomainVec[domIdx].stateHistVec[innerStep + 1] = subdomainVec[domIdx].state;

                        // set (interpolated) boundary conditions

                        // update local step and time
                        stepDom++;
                        timeDom += dtDom;

                    } // domain loop

                    // broadcast boundary conditions
                    broadcast_bcState(domIdx);

                }

                // check convergence for all domains, break if conditions met
                double abs_err = 0.0;
                double rel_err = 0.0;
                for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {
                    abs_err += convergeVals[domIdx][0];
                    rel_err += convergeVals[domIdx][1];
                }
                abs_err /= m_ndomains;
                rel_err /= m_ndomains;
                cout << "Average abs err: " << abs_err << endl;
                cout << "Average rel err: " << rel_err << endl;
                if ((rel_err < rel_err_tol) || (abs_err < abs_err_tol)) {
                    break;
                }

                convergeStep++;

                // reset interior state if not converged
                for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {
                    subdomainVec[domIdx].state = subdomainVec[domIdx].stateHistVec[0];
                }

            } // convergence loop

        }


    public:

        int m_ndomains;
        double dtMax;
        vector<subdom_t> subdomainVec;

    private:

        // mesh decomposition
        int m_dim;
        int m_ndomX = 1;
        int m_ndomY = 1;
        int m_ndomZ = 1;
        int m_overlap;
        int m_dofPerCell;

        // subdomain communication
        int bcStencilSize;
        vector<vector<int>> exchDomIdVec;
        vector<graph_t> exchGraphVec;
        vector<graph_t> ghostGraphVec;

        // time-stepping
        vector<double> dt;
        vector<int> controlItersVec;

  };
}

#endif