import os
from argparse import ArgumentParser, RawTextHelpFormatter

import numpy as np

from pdas.data_utils import load_info_domain, load_mesh_single


def make_subdomain_list(ndom_list, default=None):
    return [[[default for _ in range(ndom_list[2])] for _ in range(ndom_list[1])] for _ in range(ndom_list[0])]

# def get_cell_idxs()


# a lot of this is copied from pressio-demoapps "create_sample_mesh.py"
# have to recreate it, since it doesn't account for Schwarz overlap cells in stencil mesh
def main(decompdir, sampdir, gidfile):

    # read decomposition information
    ndom_list, overlap = load_info_domain(decompdir)
    ndomains = np.prod(ndom_list)

    # load info for each subdomain first
    coords_sub = make_subdomain_list(ndom_list)
    connect_sub = make_subdomain_list(ndom_list)
    samp_gids_sub = make_subdomain_list(ndom_list)
    stencil_gids_sub = make_subdomain_list(ndom_list)
    dims_sub = make_subdomain_list(ndom_list)
    dcell_sub = make_subdomain_list(ndom_list)
    for dom_idx in range(ndomains):
        i = dom_idx % ndom_list[0]
        j = int(dom_idx / ndom_list[0])
        k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

        # read mesh info
        meshdir_sub = os.path.join(decompdir, f"domain_{dom_idx}")
        coords_sub[i][j][k] = load_mesh_single(meshdir_sub)
        ndim = coords_sub[i][j][k].shape[-1]
        dims_sub[i][j][k] = coords_sub[i][j][k].shape[:-1]
        if ndim == 1:
            dims_sub[i][j][k] += (0, 0)
        elif ndim == 2:
            dims_sub[i][j][k] += (0,)
        connect_sub[i][j][k] = np.loadtxt(os.path.join(meshdir_sub, "connectivity.dat"), dtype=np.int32)[:, 1:]

        # get dx, dy, dz
        dcell_sub[i][j][k] = [None for _ in range(ndim)]
        with open(os.path.join(meshdir_sub, "info.dat"), "r") as f:
            for line in f.readlines():
                if "dx" in line:
                    val = float(line.strip().split(" ")[1])
                    dcell_sub[i][j][k][0] = val
                if "dy" in line:
                    val = float(line.strip().split(" ")[1])
                    dcell_sub[i][j][k][1] = val
                if "dz" in line:
                    val = float(line.strip().split(" ")[1])
                    dcell_sub[i][j][k][2] = val

        # read sample GIDs
        samp_gid_file = os.path.join(sampdir, f"domain_{dom_idx}", gidfile)
        samp_gids_sub[i][j][k] = np.sort(np.loadtxt(samp_gid_file, dtype=np.int32))

        # get local stencil GIDs (excluding ghost cells)
        stencil_gids_sub[i][j][k] = samp_gids_sub[i][j][k].copy()
        connect_samp = np.ravel(connect_sub[i][j][k][samp_gids_sub[i][j][k], :])
        connect_samp = np.delete(connect_samp, np.where(connect_samp == -1))
        stencil_gids_sub[i][j][k] = np.unique(np.concatenate((stencil_gids_sub[i][j][k], connect_samp)))

    # include stencil GIDs of ghost cells in other subdomains
    connect_samp_neigh = make_subdomain_list(ndom_list)
    stencil_gids_neigh = make_subdomain_list(ndom_list, default=[])
    for dom_idx in range(ndomains):
        i = dom_idx % ndom_list[0]
        j = int(dom_idx / ndom_list[0])
        k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

        samp_gids = samp_gids_sub[i][j][k]
        coords = coords_sub[i][j][k]
        connect_samp = connect_sub[i][j][k][samp_gids, :]
        connect_samp_neigh[i][j][k] = -1 * np.ones(connect_samp.shape, dtype=np.int32)

        nsamps = samp_gids.shape[0]
        dims = dims_sub[i][j][k]
        nstencil = int(connect_samp.shape[-1] / (2 * ndim))

        # get each direction of stencil
        for samp_idx in range(nsamps):
            samp_gid = samp_gids[samp_idx]

            x_idx = samp_gid % dims[0]
            if ndim > 1:
                y_idx = int(samp_gid / dims[0])
            if ndim == 3:
                z_idx = int(samp_gid / (dims[0] * dims[1]))

            stencil_gids = connect_samp[samp_idx, :]

            for stencil_idx in range(nstencil):
                for axis_idx in range(ndim * 2):

                    connect_idx = stencil_idx * ndim * 2 + axis_idx
                    stencil_gid = stencil_gids[connect_idx]

                    if stencil_gid == -1:

                        neigh_gid = -1
                        i_neigh = i
                        j_neigh = j
                        k_neigh = k

                        # left boundary
                        if (axis_idx == 0) and (i != 0):
                            i_neigh -= 1
                            dims_neigh = dims_sub[i-1][j][k]
                            dist = x_idx
                            neigh_gid = (dims_neigh[0] * (y_idx + 1)) - overlap - stencil_idx + dist - 1

                        # right boundary (1D)
                        if ndim == 1:
                            if (axis_idx == 1) and (i != ndom_list[0] - 1):
                                i_neigh += 1
                                dims_neigh = dims_sub[i+1][j][k]
                                dist = dims_neigh - x_idx - 1
                                neigh_gid =  overlap + stencil_idx - dist

                        if ndim > 1:

                            # front boundary
                            if (axis_idx == 1) and (j != ndom_list[1] - 1):
                                j_neigh += 1
                                dims_neigh = dims_sub[i][j+1][k]
                                dist = dims_neigh[1] - y_idx - 1
                                neigh_gid = (overlap + stencil_idx - dist) * dims_neigh[0] + x_idx

                            # right boundary (2D)
                            if (axis_idx == 2) and (i != ndom_list[0] - 1):
                                i_neigh += 1
                                dims_neigh = dims_sub[i+1][j][k]
                                dist = dims_neigh[0] - x_idx - 1
                                neigh_gid = (dims_neigh[0] * y_idx) + overlap + stencil_idx - dist

                            # back boundary
                            if (axis_idx == 3) and (j != 0):
                                j_neigh -= 1
                                dims_neigh = dims_sub[i][j-1][k]
                                dist = y_idx
                                neigh_gid = (dims_neigh[1] - 1 - overlap - stencil_idx + dist) * dims_neigh[0] + x_idx

                        if ndim == 3:
                            raise ValueError("3D not completed")
                            # bottom boundary
                            if (axis_idx == 4) and (k != 0):
                                pass
                            # top boundary
                            if (axis_idx == 5) and (k != ndom_list[2] - 2):
                                pass

                        if neigh_gid != -1:
                            stencil_gids_neigh[i][j][k].append(neigh_gid)
                            connect_samp_neigh[i][j][k][samp_idx, connect_idx] = neigh_gid
                            stencil_gids_sub[i_neigh][j_neigh][k_neigh] = np.unique(np.concatenate((stencil_gids_sub[i_neigh][j_neigh][k_neigh], [neigh_gid])))

    # generate global-to-stencil map for each subdomain
    global_to_stencil_map_sub = make_subdomain_list(ndom_list)
    for dom_idx in range(ndomains):
        i = dom_idx % ndom_list[0]
        j = int(dom_idx / ndom_list[0])
        k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

        stencil_gids = stencil_gids_sub[i][j][k]
        ncells = connect_sub[i][j][k].shape[0]
        global_to_stencil_map_sub[i][j][k] = -1 * np.ones(ncells, dtype=np.int32)

        stencil_idx = 0
        for cell_idx in range(ncells):
            if cell_idx in stencil_gids:
                global_to_stencil_map_sub[i][j][k][cell_idx] = stencil_idx
                stencil_idx += 1

    # write coordinates, map connectivity to stencil mesh IDs
    for dom_idx in range(ndomains):
        i = dom_idx % ndom_list[0]
        j = int(dom_idx / ndom_list[0])
        k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

        samp_gids = samp_gids_sub[i][j][k]
        stencil_gids = stencil_gids_sub[i][j][k]
        connect = connect_sub[i][j][k]

        ncells = connect.shape[0]

        global_to_stencil_map = global_to_stencil_map_sub[i][j][k]
        samp_gids = samp_gids_sub[i][j][k]
        connect_samp = connect[samp_gids, :]
        nsamps = samp_gids.shape[0]
        nstencil = connect_samp.shape[1]
        connect_samp_local = np.zeros((nsamps, nstencil + 1), dtype=np.int32)
        for connect_idx in range(nsamps):
            connect_samp_local[connect_idx, 0] = global_to_stencil_map[samp_gids[connect_idx]]
            for stencil_idx in range( nstencil):
                stencil_gid = connect_samp[connect_idx, stencil_idx]
                if stencil_gid == -1:
                    connect_samp_local[connect_idx, stencil_idx+1] = -1
                else:
                    connect_samp_local[connect_idx, stencil_idx+1] = global_to_stencil_map[stencil_gid]

        meshdir_sub = os.path.join(sampdir, f"domain_{dom_idx}")

        # write connectivity
        connect_file = os.path.join(meshdir_sub, "connectivity.dat")
        np.savetxt(connect_file, connect_samp_local, fmt="%8d")

        # write coordinates
        coords = np.reshape(coords_sub[i][j][k], (-1, ndim), order="F")
        with open(os.path.join(meshdir_sub, "coordinates.dat"), "w") as f:
            for stencil_idx in range(stencil_gids.shape[0]):
                coords_cell = coords[stencil_gids[stencil_idx], :]
                if ndim == 1:
                    f.write(f"{stencil_idx:8d} {coords_cell[0]:.14f}\n")
                elif ndim == 2:
                    f.write(f"{stencil_idx:8d} {coords_cell[0]:.14f} {coords_cell[1]:.14f}\n")
                elif ndim == 3:
                    f.write(f"{stencil_idx:8d} {coords_cell[0]:.14f} {coords_cell[1]:.14f} {coords_cell[2]:.14f}\n")

        # write info
        with open(os.path.join(meshdir_sub, "info.dat"), "w") as f:
            f.write(f"dim {ndim}\n")

            dx = dcell_sub[i][j][k][0]
            xMin = np.amin(coords[:, 0])
            xMax = np.amax(coords[:, 0])
            f.write(f"xMin {xMin - dx / 2:.14f}\n")
            f.write(f"xMax {xMax + dx / 2:.14f}\n")
            if ndim > 1:
                dy = dcell_sub[i][j][k][1]
                yMin = np.amin(coords[:, 1])
                yMax = np.amax(coords[:, 1])
                f.write(f"yMin {yMin - dy / 2:.14f}\n")
                f.write(f"yMax {yMax + dy / 2:.14f}\n")
            if ndim == 3:
                dz = dcell_sub[i][j][k][2]
                zMin = np.amin(coords[:, 2])
                zMax = np.amax(coords[:, 2])
                f.write(f"zMin {zMin - dz / 2:.14f}\n")
                f.write(f"zMax {zMax + dz / 2:.14f}\n")

            f.write(f"dx {dx:.14f}\n")
            if ndim > 1:
                f.write(f"dy {dcell_sub[i][j][k][1]:.14f}\n")
            if ndim == 3:
                f.write(f"dx {dcell_sub[i][j][k][2]:.14f}\n")

            f.write(f"sampleMeshSize {nsamps:8d}\n")
            f.write(f"stencilMeshSize {stencil_gids.shape[0]:8d}\n")
            stencil_size = int(connect_samp.shape[-1] / ndim + 1)
            f.write(f"stencilSize {stencil_size:2d}\n")

        # write neighboring stencil IDs
        with open(os.path.join(meshdir_sub, "connectivity_neighbor.dat"), "w") as f:
            for samp_idx in range(nsamps):
                if any(connect_samp_neigh[i][j][k][samp_idx, :] != -1):
                    stencil_id = global_to_stencil_map[samp_gids[samp_idx]]
                    f.write(f"{stencil_id:8d}")
                    for stencil_idx in range(nstencil):
                        neigh_gid = connect_samp_neigh[i][j][k][samp_idx, stencil_idx]
                        if neigh_gid == -1:
                            neigh_sid = -1
                        else:
                            i_neigh = i
                            j_neigh = j
                            k_neigh = k
                            remain = stencil_idx % (ndim * 2)
                            # left
                            if remain == 0:
                                i_neigh -= 1
                            # right (1D)
                            if (ndim == 1) and (remain == 1):
                                i_neigh += 1
                            if ndim > 1:
                                # front
                                if remain == 1:
                                    j_neigh += 1
                                # right
                                if remain == 2:
                                    i_neigh += 1
                                # back
                                if remain == 3:
                                    j_neigh -= 1
                            if ndim == 3:
                                # bottom
                                if remain == 4:
                                    k_neigh -= 1
                                # top
                                if remain == 5:
                                    k_neigh += 1
                            global_to_stencil_map_neigh = global_to_stencil_map_sub[i_neigh][j_neigh][k_neigh]
                            neigh_sid = global_to_stencil_map_neigh[neigh_gid]

                        f.write(f" {neigh_sid}")
                    f.write("\n")

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    # location of full decomposed mesh
    parser.add_argument(
        "--decompMeshDir", "--decompmeshdir", "--decomp_mesh_dir",
        dest="decompdir",
        help="Full path to base directory of decomposed mesh.",
    )

    # location of sample mesh base directory (which already has sample mesh GIDs)
    parser.add_argument(
        "--sampleMeshDir", "--samplemeshdir", "--sample_mesh_dir",
        dest="sampdir",
        help="Full path to base directory of decomposed sample mesh.",
    )

    # name of sample mesh GIDs file (optional)
    parser.add_argument(
        "--sampleGIDFile", "--sample_gid_file",
        dest="gidfile",
        default="sample_mesh_gids.txt",
        help="Name of sample mesh GID file.",
    )

    argobj = parser.parse_args()

    main(argobj.decompdir, argobj.sampdir, argobj.gidfile)