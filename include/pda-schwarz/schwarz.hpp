
#ifndef PRESSIODEMOAPPS_SCHWARZ_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_HPP_

#include "BS_thread_pool.hpp"
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
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include <filesystem>


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


enum class SchwarzMode{ Multiplicative, Additive };

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

        // silly, but some things have to be written to disk for hyper-reduction,
        //      as mesh class HAS to be instantiated from a mesh directory
        m_tempdir = "./temp_" + std::to_string(::getpid());
        std::filesystem::create_directory(m_tempdir);

        // set up connectivity for neighboring subdomains
        // only relevant if at least one domain is a hyper-reduction subdomain
        calc_hyper_connectivity();

        // hyper-reduction subdomains need some final member object initializations
        // this is a consequence of computing the stencil mesh at runtime
        for (int domIdx = 0; domIdx < m_subdomainVec.size(); ++domIdx) {
            m_subdomainVec[domIdx]->finalize_subdomain(m_tempdir);
        }

        setup_controller(dtVec);
        for (int domIdx = 0; domIdx < m_subdomainVec.size(); ++domIdx) {
            m_subdomainVec[domIdx]->allocateStorageForHistory(m_controlItersVec[domIdx]);
        }

        // set up communication patterns, first communication
        calc_exch_graph();
        for (int domIdx = 0; domIdx < m_subdomainVec.size(); ++domIdx) {
            broadcast_bcState(domIdx);
        }

        // set up ghost filling graph, boundary pointers
        calc_ghost_graph();

        // delete temporary directory
        std::filesystem::remove_all(m_tempdir);

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

    int grid_to_linear_idx(int i, int j, int k) {
        const auto & tiling = *m_tiling;
        return i + j * tiling.countX() + k * tiling.countX() * tiling.countY();
    }

    auto linear_to_grid_idx(int domIdx) {
        const auto & tiling = *m_tiling;
        int i = domIdx % tiling.countX();
        int j = domIdx / tiling.countX();
        int k = domIdx / (tiling.countX() * tiling.countY());
        return std::tuple(i, j, k);
    }

    void calc_hyper_connectivity()
    {
        const auto & tiling = *m_tiling;
        int overlap = tiling.overlap();

        // various storage required
        std::vector<std::vector<int>> stencil_gids;
        std::vector<graph_t> neigh_gids;
        std::vector<std::vector<int>> global_to_stencil_map;
        stencil_gids.resize(tiling.count());
        neigh_gids.resize(tiling.count());
        global_to_stencil_map.resize(tiling.count());

        // get stencil GIDs from each subdomain
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {
            auto [i, j, k] = linear_to_grid_idx(domIdx);

            auto & meshFull = m_subdomainVec[domIdx]->getMeshFull();
            auto * sampGids = m_subdomainVec[domIdx]->getSampleGids();
            const auto & graphFull = meshFull.graph();

            for (int sampIdx = 0; sampIdx < sampGids->rows(); ++sampIdx) {
                int samp_gid = (*sampGids)(sampIdx);
                auto graph_row = graphFull.row(samp_gid);
                for (int stencilIdx = 0; stencilIdx < graph_row.cols(); ++stencilIdx) {
                    int stencil_gid = graph_row(0, stencilIdx);
                    if (stencil_gid != -1) {
                        stencil_gids[domIdx].emplace_back(stencil_gid);
                    }
                }
            }
        }

        // get stencil GIDs that are required from neighboring domain
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {
            auto [i, j, k] = linear_to_grid_idx(domIdx);

            const auto & meshFull = m_subdomainVec[domIdx]->getMeshFull();
            std::array<int, 3> fullMeshDims = m_subdomainVec[domIdx]->getFullMeshDims();
            const auto * sampGids = m_subdomainVec[domIdx]->getSampleGids();
            const auto & graphFull = meshFull.graph();

            pda::resize(neigh_gids[domIdx], sampGids->rows(), graphFull.cols());
            neigh_gids[domIdx].setConstant(-1);

            int x_idx = 0;
            int y_idx = 0;
            int z_idx = 0;
            int dist, neighIdx;
            for (int sampIdx = 0; sampIdx < sampGids->rows(); ++sampIdx) {
                int samp_gid = (*sampGids)(sampIdx);
                auto graph_row = graphFull.row(samp_gid);
                neigh_gids[domIdx](sampIdx, 0) = samp_gid;

                x_idx = samp_gid % fullMeshDims[0];
                if (tiling.dim() > 1) {
                    y_idx = samp_gid / fullMeshDims[0];
                }
                if (tiling.dim() == 3) {
                    z_idx = samp_gid / (fullMeshDims[0] * fullMeshDims[1]);
                }

                int nstencil_1d = (graph_row.cols() - 1) / (tiling.dim() * 2);
                for (int axisIdx = 0; axisIdx < tiling.dim() * 2; ++axisIdx) {
                    for (int stencilIdx = 0; stencilIdx < nstencil_1d; ++stencilIdx) {

                        int connect_idx = stencilIdx * tiling.dim() * 2 + axisIdx + 1;
                        int stencil_gid = graph_row(0, connect_idx);

                        if (stencil_gid == -1) {
                            int neigh_gid = -1;
                            int i_neigh = i;
                            int j_neigh = j;
                            int k_neigh = k;

                            // left boundary
                            if ((axisIdx == 0) && (i != 0)) {
                                i_neigh -= 1;
                                neighIdx = grid_to_linear_idx(i-1, j, k);
                                auto dims_neigh = m_subdomainVec[neighIdx]->getFullMeshDims();
                                dist = x_idx;
                                neigh_gid = (dims_neigh[0] * (y_idx + 1)) - overlap - stencilIdx + dist - 1;
                            }

                            // right boundary (1D)
                            if (tiling.dim() == 1) {
                                if ((axisIdx == 1) && (i != tiling.countX() - 1)) {
                                    i_neigh += 1;
                                    neighIdx = grid_to_linear_idx(i+1, j, k);
                                    auto dims_neigh = m_subdomainVec[neighIdx]->getFullMeshDims();
                                    dist = dims_neigh[0] - x_idx - 1;
                                    neigh_gid =  overlap + stencilIdx - dist;
                                }
                            }

                            if (tiling.dim() > 1) {

                                // front boundary
                                if ((axisIdx == 1) && (j != tiling.countY() - 1)) {
                                    j_neigh += 1;
                                    neighIdx = grid_to_linear_idx(i, j+1, k);
                                    auto dims_neigh = m_subdomainVec[neighIdx]->getFullMeshDims();
                                    dist = dims_neigh[1] - y_idx - 1;
                                    neigh_gid = (overlap + stencilIdx - dist) * dims_neigh[0] + x_idx;
                                }

                                // right boundary (2D)
                                if ((axisIdx == 2) && (i != tiling.countX() - 1)) {
                                    i_neigh += 1;
                                    neighIdx = grid_to_linear_idx(i+1, j, k);
                                    auto dims_neigh = m_subdomainVec[neighIdx]->getFullMeshDims();
                                    dist = dims_neigh[0] - x_idx - 1;
                                    neigh_gid = (dims_neigh[0] * y_idx) + overlap + stencilIdx - dist;
                                }

                                // back boundary
                                if ((axisIdx == 3) && (j != 0)) {
                                    j_neigh -= 1;
                                    neighIdx = grid_to_linear_idx(i, j-1, k);
                                    auto dims_neigh = m_subdomainVec[neighIdx]->getFullMeshDims();
                                    dist = y_idx;
                                    neigh_gid = (dims_neigh[1] - 1 - overlap - stencilIdx + dist) * dims_neigh[0] + x_idx;
                                }
                            }

                            if (tiling.dim() == 3) {
                                throw std::runtime_error("3D not implemented yet");
                            }

                            if (neigh_gid != -1) {
                                neigh_gids[domIdx](sampIdx, connect_idx) = neigh_gid;
                                stencil_gids[neighIdx].emplace_back(neigh_gid);
                            }
                        }
                    }
                }
            }
        }

        // sort and store stencil GIDs
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {
            auto [i, j, k] = linear_to_grid_idx(domIdx);
            std::sort( stencil_gids[domIdx].begin(), stencil_gids[domIdx].end() );
            stencil_gids[domIdx].erase(
                std::unique(
                    stencil_gids[domIdx].begin(),
                    stencil_gids[domIdx].end()
                ),
                stencil_gids[domIdx].end()
            );

            m_subdomainVec[domIdx]->setStencilGids(stencil_gids[domIdx]);

        }

        // generate global-to-stencil map
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {
            auto [i, j, k] = linear_to_grid_idx(domIdx);
            const auto & meshFull = m_subdomainVec[domIdx]->getMeshFull();

            global_to_stencil_map[domIdx].resize(meshFull.sampleMeshSize(), -1);
            int stencilIdx = 0;
            for (int cellIdx = 0; cellIdx < meshFull.sampleMeshSize(); ++cellIdx) {
                if (std::find(stencil_gids[domIdx].begin(), stencil_gids[domIdx].end(), cellIdx) != stencil_gids[domIdx].end()) {
                    global_to_stencil_map[domIdx][cellIdx] = stencilIdx;
                    stencilIdx++;
                }
            }
        }

        // write coordinates and connectivity
        for (int domIdx = 0; domIdx < tiling.count(); ++domIdx) {
            // make subdirectory
            std::string subdom_dir = m_tempdir + "/domain_" + std::to_string(domIdx);
            std::filesystem::create_directory(subdom_dir);

            auto [i, j, k] = linear_to_grid_idx(domIdx);
            const auto & meshFull = m_subdomainVec[domIdx]->getMeshFull();
            const auto * sampGids = m_subdomainVec[domIdx]->getSampleGids();
            const auto & graphFull = meshFull.graph();

            // stencil GIDs
            std::ofstream stencil_file(subdom_dir + "/stencil_mesh_gids.dat");
            for (int stencilIdx = 0; stencilIdx < stencil_gids[domIdx].size(); ++stencilIdx) {
                int stencil_gid = stencil_gids[domIdx][stencilIdx];
                stencil_file << std::to_string(stencil_gid) + "\n";
            }
            stencil_file.close();

            // connectivity
            std::ofstream connect_hyper_file(subdom_dir + "/connectivity.dat");
            for (int sampIdx = 0; sampIdx < sampGids->rows(); ++sampIdx) {
                int samp_gid = (*sampGids)(sampIdx);
                connect_hyper_file << std::to_string(global_to_stencil_map[domIdx][samp_gid]);
                for (int stencilIdx = 1; stencilIdx < graphFull.cols(); ++stencilIdx) {
                    int stencil_gid = graphFull(samp_gid, stencilIdx);
                    if (stencil_gid == -1) {
                        connect_hyper_file << " " + std::to_string(-1);
                    }
                    else {
                        connect_hyper_file << " " + std::to_string(global_to_stencil_map[domIdx][stencil_gid]);
                    }
                }
                connect_hyper_file << "\n";
            }
            connect_hyper_file.close();

            // coordinates
            std::ofstream coords_file(subdom_dir + "/coordinates.dat");
            auto & xcoords = meshFull.viewX();
            auto & ycoords = meshFull.viewY();
            auto & zcoords = meshFull.viewZ();
            for (int stencilIdx = 0; stencilIdx < stencil_gids[domIdx].size(); ++stencilIdx) {
                int stencil_gid = stencil_gids[domIdx][stencilIdx];
                coords_file << std::to_string(stencilIdx);

                coords_file << " " << std::fixed << std::setprecision(14) << xcoords(stencil_gid);
                if (tiling.dim() > 1) {
                    coords_file << " " << std::fixed << std::setprecision(14) << ycoords(stencil_gid);
                }
                if (tiling.dim() == 3) {
                    coords_file << " " << std::fixed << std::setprecision(14) << zcoords(stencil_gid);
                }
                coords_file << "\n";
            }
            coords_file.close();

            // info
            std::ofstream info_file(subdom_dir + "/info.dat");
            info_file << "dim " + std::to_string(tiling.dim()) + "\n";
            auto xmin = xcoords.minCoeff() - meshFull.dx() / 2.0;
            auto xmax = xcoords.maxCoeff() + meshFull.dx() / 2.0;
            info_file << "xMin " << std::fixed << std::setprecision(14) << xmin << "\n";
            info_file << "xMax " << std::fixed << std::setprecision(14) << xmax << "\n";
            if (tiling.dim() > 1) {
                auto ymin = ycoords.minCoeff() - meshFull.dy() / 2.0;
                auto ymax = ycoords.maxCoeff() + meshFull.dy() / 2.0;
                info_file << "yMin " << std::fixed << std::setprecision(14) << ymin << "\n";
                info_file << "yMax " << std::fixed << std::setprecision(14) << ymax << "\n";
            }
            if (tiling.dim() == 3) {
                auto zmin = zcoords.minCoeff() - meshFull.dz() / 2.0;
                auto zmax = zcoords.maxCoeff() + meshFull.dz() / 2.0;
                info_file << "zMin " << std::fixed << std::setprecision(14) << zmin << "\n";
                info_file << "zMax " << std::fixed << std::setprecision(14) << zmax << "\n";
            }
            info_file << "dx " << std::fixed << std::setprecision(14) << meshFull.dx() << "\n";
            if (tiling.dim() > 1) {
                info_file << "dy " << std::fixed << std::setprecision(14) << meshFull.dy() << "\n";
            }
            if (tiling.dim() == 3) {
                info_file << "dz " << std::fixed << std::setprecision(14) << meshFull.dz() << "\n";
            }
            info_file << "sampleMeshSize " << sampGids->rows() << "\n";
            info_file << "stencilMeshSize " << stencil_gids[domIdx].size() << "\n";
            info_file << "stencilSize " << meshFull.stencilSize() << "\n";
            info_file.close();

            // generate sample mesh (noop for FOM/PROM)
            m_subdomainVec[domIdx]->genHyperMesh(subdom_dir);

            // generate neighbor connectivity
            graph_t neighborGraph;
            pda::resize(neighborGraph, sampGids->rows(), graphFull.cols());
            for (int sampIdx = 0; sampIdx < sampGids->rows(); ++sampIdx) {
                int samp_gid = (*sampGids)(sampIdx);
                int stencil_gid = global_to_stencil_map[domIdx][samp_gid];
                neighborGraph(sampIdx, 0) = stencil_gid;
                for (int stencilIdx = 1; stencilIdx < graphFull.cols(); ++stencilIdx) {
                    int neigh_gid = neigh_gids[domIdx](sampIdx, stencilIdx);
                    int neigh_sid;
                    if (neigh_gid == -1) {
                        neigh_sid = -1;
                    }
                    else {
                        int i_neigh = i;
                        int j_neigh = j;
                        int k_neigh = k;
                        int remain = (stencilIdx - 1) % (tiling.dim() * 2);

                        // left
                        if (remain == 0) {
                            i_neigh -= 1;
                        }
                        // right (1D)
                        if ((tiling.dim() == 1) and (remain == 1)) {
                            i_neigh += 1;
                        }
                        if (tiling.dim() > 1) {
                            // front
                            if (remain == 1) {
                                j_neigh += 1;
                            }
                            // right
                            if (remain == 2) {
                                i_neigh += 1;
                            }
                            // back
                            if (remain == 3) {
                                j_neigh -= 1;
                            }
                        }
                        if (tiling.dim() == 3) {
                            // bottom
                            if (remain == 4) {
                                k_neigh -= 1;
                            }
                            // top
                            if (remain == 5) {
                                k_neigh += 1;
                            }
                        }
                        int neighIdx = grid_to_linear_idx(i_neigh, j_neigh, k_neigh);
                        neigh_sid = global_to_stencil_map[neighIdx][neigh_gid];
                    }
                    neighborGraph(sampIdx, stencilIdx) = neigh_sid;
                }
            }

            m_subdomainVec[domIdx]->setNeighborGraph(neighborGraph);

        }

    }

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

        if (m_tiling->dim() == 1) {
            if ((id1 == 0) && (id2 == 1)) { return true; }
            else { return false; }
        }
        else if (m_tiling->dim() >= 2) {
            if ((id1 == 0) && (id2 == 2)) { return true; }
            else if ((id1 == 1) && (id2 == 3)) { return true; }

            if (m_tiling->dim() == 3) {
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
    [[nodiscard]] std::vector<std::vector<double>> calc_controller_step(
        SchwarzMode mode,
        int outerStep,
        double currentTime,
        const double rel_err_tol,
        const double abs_err_tol,
        const int convergeStepMax)
    {
      return calc_controller_step_impl(outerStep, currentTime,  rel_err_tol, abs_err_tol,
				       convergeStepMax, mode, nullptr);
    }

    [[nodiscard]] std::vector<std::vector<double>> calc_controller_step(
        SchwarzMode mode,
        int outerStep,
        double currentTime,
        const double rel_err_tol,
        const double abs_err_tol,
        const int convergeStepMax,
	BS::thread_pool & pool)
    {
      const bool is_multipl = mode==SchwarzMode::Multiplicative;
      if (is_multipl && pool.get_thread_count() != 1){
	throw std::runtime_error("you cannot run multiplicative Schwarz in parallel (yet), reset the # of thread to 1");
      }
      return calc_controller_step_impl(outerStep, currentTime,  rel_err_tol, abs_err_tol,
				       convergeStepMax, mode, &pool);
    }

private:
    [[nodiscard]] std::vector<std::vector<double>> calc_controller_step_impl(
        int outerStep,
        double currentTime,
        const double rel_err_tol,
        const double abs_err_tol,
        const int convergeStepMax,
        SchwarzMode mode,
        BS::thread_pool * pool)
    {
        const bool additive = (mode==SchwarzMode::Additive);
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

	    std::cout << "Schwarz iteration " << convergeStep + 1 << '\n';

	    auto maintask = [&, currentTime, outerStep](const int domIdx){
                // reset to beginning of controller time
                auto timeDom = currentTime;
                auto stepDom = outerStep * m_controlItersVec[domIdx];
                const auto dtDom = m_dt[domIdx];
                const auto dtWrap = pode::StepSize<double>(dtDom);
                auto runtimeStart = std::chrono::high_resolution_clock::now();

                for (int innerStep = 0; innerStep < m_controlItersVec[domIdx]; ++innerStep) {
                    const auto startTimeWrap = pode::StepStartAt<double>(timeDom);
                    const auto stepWrap = pode::StepCount(stepDom);
                    m_subdomainVec[domIdx]->doStep(startTimeWrap, stepWrap, dtWrap);
                    m_subdomainVec[domIdx]->updateFullState(); // noop for FOM subdomain

                    // for last iteration, compute convergence criteria
                    // important to do this before saving history,
		    // as stateHistVec still has last convergence loop's state
                    // NOTE: this is always computed on the full-order state
                    if (innerStep == (m_controlItersVec[domIdx] - 1)) {
                        convergeVals[domIdx] = calcConvergence(
                               *m_subdomainVec[domIdx]->getStateStencil(),
                               m_subdomainVec[domIdx]->getLastStateInHistory());
                    }

                    // store intra-step history
                    m_subdomainVec[domIdx]->storeStateHistory(innerStep+1);

                    // set (interpolated) boundary conditions

                    stepDom++;
                    timeDom += dtDom;
                }

                // broadcast boundary conditions immediately for multiplicative Schwarz
                if (!additive) { broadcast_bcState(domIdx); }

                // record iteration runtime (in seconds)
                auto runtimeEnd = std::chrono::high_resolution_clock::now();
                double nsElapsed = static_cast<double>
		  (std::chrono::duration_cast<std::chrono::nanoseconds>(runtimeEnd - runtimeStart).count());
                iterTime[domIdx].emplace_back(nsElapsed * 1e-9);
            };

	    if (pool){
	      pool->detach_loop<int>(0, ndomains, maintask);
	      pool->wait();
	    }
	    else{
	      for (int domIdx = 0; domIdx < ndomains; ++domIdx) {
		maintask(domIdx);
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
            std::cout << "Average abs err: " << abs_err << '\n';
            std::cout << "Average rel err: " << rel_err << '\n';
            if ((rel_err < rel_err_tol) || (abs_err < abs_err_tol)) {
                break;
            }

            // broadcast boundary conditions after domain cycle for additive Schwarz
            if (additive) {
	      auto task = [&](const int domIdx){ broadcast_bcState(domIdx); };
	      if (pool){
		pool->detach_loop<int>(0, ndomains, task);
		pool->wait();
	      }
	      else{
		for (int domIdx = 0; domIdx < ndomains; ++domIdx) { task(domIdx); }
	      }
	    }
            convergeStep++;

	    // reset interior state if not converged
	    auto taskreset = [&](const int domIdx){ m_subdomainVec[domIdx]->resetStateFromHistory(); };
	    if (pool){
	      pool->detach_loop<int>(0, ndomains, taskreset);
	      pool->wait();
	    }
	    else{
	      for (int domIdx = 0; domIdx < ndomains; ++domIdx){ taskreset(domIdx); }
	    }

        } // convergence loop

        return iterTime;
    }

public:

    std::string m_tempdir;
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
