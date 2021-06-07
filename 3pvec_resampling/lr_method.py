import numpy as np
from numpy.lib.arraysetops import isin 
from scipy import sparse
import scipy.linalg as spla 
import copy
import regtricks as rt 
from pdb import set_trace
from toblerone import utils
import fabber_funcs
import tempfile 
import os.path as op 
import os 
import nibabel
import multiprocessing 
import functools


def _calc_volumetric_adjacency(mask, vox_size=np.ones(3), distance_weight=1):

    cardinal_neighbours = np.array([
        [-1,0,0],[1,0,0],[0,-1,0],
        [0,1,0],[0,0,-1],[0,0,1]], 
        dtype=np.int32)
    dist_weight = np.array((vox_size, vox_size)).T.flatten()
    dist_weight = 1 / (dist_weight ** distance_weight)

    # Generate a Vx3 array of all voxel indices (IJK, 000, 001 etc)
    # Add the vector of cardinal numbers to each to get the neighbourhood
    # Mask out negative indices or those outside the mask shape. 
    # Test is applied along final dimension to give a Vx6 bool mask, the 
    # 6 neighbours of each voxel. 
    vox_ijk = np.moveaxis(np.indices(mask.shape), 0, 3).reshape(-1,3)
    vox_neighbours = vox_ijk[:,None,:] + cardinal_neighbours[None,:,:]
    in_grid = ((vox_neighbours >= 0) & (vox_neighbours < mask.shape)).all(-1)

    # Flatnonzero and mod division of the mask array gives us numbers in 
    # the range [0,5] for each accepted neighbour in the mask. These numbers
    # correspond to X,Y,Z in the distance weights, so this gives us the 
    # entries for the final adjacency matrix 
    xyz = np.mod(np.flatnonzero(in_grid), 6)
    data = dist_weight[xyz]

    # Column indices are found by converting IJK indices of each accepted
    # neighbour into flat indices 
    # Indptr: see sparse.csr_matrix documentation, must be of size rows+1, 
    # where the first entry is always 0. The entries are found by taking
    # the cumulative sum of the number of accepted neighbours per voxel, ie, 
    # sum along columns of the mask. 
    col = np.ravel_multi_index(vox_neighbours[in_grid,:].T, mask.shape)
    indptr = np.concatenate(([0], np.cumsum(in_grid.sum(1))))

    # Form CSR matrix 
    size = np.prod(mask.shape)
    mat = sparse.csr_matrix((data, col, indptr), shape=(size, size))    

    return mat


def lr_separation(data, mask, pvs):

    mask = nibabel.load(mask).get_fdata().astype(np.bool)
    adj = _calc_volumetric_adjacency(mask)

    data = nibabel.load(data).get_fdata()
    data = np.moveaxis(data, 3, 0)
    tpts = data.shape[0]

    if pvs.shape[3] > 2: 
        pvs = pvs[...,:2]

    # corrected = np.zeros((mask.sum(), tpts, 2))
    pvs = pvs.reshape(-1,2)
    data = data.reshape(tpts,-1)
    assert mask.size == pvs.shape[0]

    worker = functools.partial(_separator_worker, mask=mask, adj=adj, pvs=pvs)
    with multiprocessing.Pool() as p: 
        vols = p.map(worker, data)

    vols = np.stack(vols, axis=-1)
    data_c = np.zeros((*mask.shape, 2, tpts))
    data_c[mask,...] = vols 
    data_c = np.swapaxes(data_c, 3, 4)
    return data_c


def _separator_worker(data_vol, mask, adj, pvs):

    corrected = np.zeros((mask.sum(), 2))
    for idx,vidx in enumerate(np.flatnonzero(mask)):
        neighbours = adj[vidx,:].indices
        w = pvs[neighbours,:]
        y = data_vol[neighbours]
        cbf = np.linalg.lstsq(w, y)[0]
        corrected[idx,:] = cbf

    return corrected 



def oxasl_lr(data, calib, mask, odir, basil_opts, pvs):
    
    data_paths = [] 
    for tiss in ['gm', 'wm']:
        d = op.join(odir, tiss)
        os.makedirs(d, exist_ok=True)
        data_paths.append(op.join(d, 'data.nii.gz'))
    
    pvs = np.stack([ nibabel.load(p).get_fdata() for p in pvs ], axis=-1)
    spc = rt.ImageSpace(mask)

    if not op.exists(data_paths[0]):
        data_c = lr_separation(data, mask, pvs)
        for idx,path in enumerate(data_paths):
            spc.save_image(data_c[...,idx], path)

    gm_opts = { **basil_opts, 'pc': 0.98, 't1': 1.3, 'fcalib': 0.01, 'bat': 1.3 }
    wm_opts = { **basil_opts, 'pc': 0.8, "t1": 1.1, 'fcalib': 0.003, 'bat': 1.6 }
    cmds = [] 
    for path, opts in zip(data_paths, [gm_opts, wm_opts]):
        cmds.append(fabber_funcs.oxasl_cmd(path, calib, mask, op.dirname(path), opts))
    
    return cmds


def basil_lr(data, mask, basil_opts, odir, pvs):

    data_paths = [] 
    for tiss in ['gm', 'wm']:
        d = op.join(odir, tiss)
        os.makedirs(d, exist_ok=True)
        data_paths.append(op.join(d, 'data.nii.gz'))
    
    if isinstance(pvs, list): 
        pvs = np.stack([nibabel.load(p).get_fdata() for p in pvs], axis=-1)
    elif isinstance(pvs, str):
        pvs = nibabel.load(pvs).get_fdata()

    spc = rt.ImageSpace(mask)

    if not op.exists(data_paths[0]):
        data_c = lr_separation(data, mask, pvs)
        for idx,path in enumerate(data_paths):
            spc.save_image(data_c[...,idx], path)

    gm_opts = { **basil_opts, 'pc': 0.98, 't1': 1.3, 'fcalib': 0.01, 'bat': 1.3 }
    wm_opts = { **basil_opts, 'pc': 0.8, "t1": 1.1, 'fcalib': 0.003, 'bat': 1.6 }
    cmds = [] 
    for path, opts in zip(data_paths, [gm_opts, wm_opts]):
        cmds.append(fabber_funcs.basil_cmd(path, mask, opts,
            op.dirname(path)))

    return cmds 