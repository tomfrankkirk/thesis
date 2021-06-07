# Utilities used throughout the toblerone evaluation scripts

# This script requires that the HCP's wb_command be installed 

import numpy as np 
import os.path as op 
import subprocess 
import toblerone as t 
import nibabel
from toblerone.classes import ImageSpace
from toblerone.utils import STRUCTURES
import glob
from toblerone.main import stack_images
import copy 
import pdb


def make_reference(orig, FoV, v, fname):
    orig = np.array([90, -126, -72])

    dims = np.ceil(FoV / v)
    cmd = 'wb_command -volume-create %d %d %d ' % (dims[0], dims[1], dims[2])
    cmd += '%s -sform -%.3f 0 0 %.3f '  % (fname, v, orig[0])
    cmd += '0 %.3f 0 %.3f ' % (v, orig[1])
    cmd += '0 0 %.3f %.3f ' % (v, orig[2])

    subprocess.run(cmd, shell=True)


def restack(dirname, suff):
    keys = STRUCTURES + ['FAST_CSF', 'FAST_GM', 'FAST_WM', 'cortex_GM', 'cortex_WM', 'cortex_nonbrain']
    keys.remove('BrStem')
    if not suff: 
        paths = [ glob.glob(op.join(dirname, '*%s*' % k))[0] for k in keys ]
    else: 
        paths = [ glob.glob(op.join(dirname, '*%s%s*' % (k,suff)))[0] for k in keys ]
    assert len(paths) == 20, 'Did not find all images for all keys'
    imgs = dict(zip(keys, [nibabel.load(p).get_fdata() for p in paths]))
    newstacked = stack_images(imgs)
    return newstacked

def do_RC(LWS,LPS,RWS,RPS,RMS,LMS,outname,vox,ref):

    od = op.dirname(outname)

    for s in [LWS,LPS,RWS,RPS]:
        if not op.isfile(s):
            raise RuntimeError("Surface %s does not exist" % s)

    sides = ['L', 'R']
    ltemp, rtemp = map(lambda s: op.join(od, '%s_template.func.gii' % s), sides) 
    lones, rones = map(lambda s: op.join(od, '%s_ones.func.gii' % s), sides) 

    for white,mid,pial,temp,ones in zip([LWS,RWS], [LMS,RMS], [LPS,RPS], 
        [ltemp,rtemp], [lones,rones]):

        # Create mid surface
        cmd = 'wb_command -surface-cortex-layer %s %s 0.5 %s' % (white, pial, mid)
        subprocess.run(cmd, shell=True)

        # Produce template metric file
        cmd = 'wb_command -surface-vertex-areas %s %s' % (white, temp)
        subprocess.run(cmd, shell=True)

        # Produce a metric file of ones using template
        cmd = 'wb_command -metric-math ''1'' %s -var x %s' % (ones, temp)
        subprocess.run(cmd, shell=True)
            
    lgm,rgm = map(lambda s: op.join(od, '%s_%1.1f_hcpgm.nii.gz' % (s, vox)), sides) 
    lwm,rwm = map(lambda s: op.join(od, '%s_%1.1f_hcpwm.nii.gz' % (s, vox)), sides) 
    lcsf,rcsf = map(lambda s: op.join(od, '%s_%1.1f_hcpcsf.nii.gz' % (s, vox)), sides) 
    lsd,rsd = map(lambda s: op.join(od, '%s_%1.1f_signdist.nii.gz' % (s, vox)), sides) 

    subdiv = np.maximum(np.ceil(vox/0.4), 4).astype(np.int8)

    for white,mid,pial,gm,wm,csf,sgndist,ones in zip([LWS,RWS], [LMS,RMS], [LPS,RPS],
        [lgm,rgm], [lwm,rwm], [lcsf,rcsf], [lsd,rsd], [lones,rones]):

        # Map metric to volume for PV fracs
        cmd = ("wb_command -metric-to-volume-mapping " + 
            "{} {} {} {} ".format(ones, mid, ref, gm) + 
            "-ribbon-constrained {} {} -voxel-subdiv {}".format(white, pial, subdiv))
        subprocess.run(cmd, shell=True)

        # Estimate WM/CSF via sign distance method
        cmd = ("wb_command -create-signed-distance-volume %s %s %s" % (mid, ref, sgndist)
            + " -approx-limit 60 -fill-value 60")
        subprocess.run(cmd, shell=True)

        # WM
        cmd = ("wb_command -volume-math '(1 - GM) * (sd < 0)'" 
            + " %s -var GM %s -var sd %s" % (wm, gm, sgndist))
        subprocess.run(cmd, shell=True)

        # CSF
        cmd = ("wb_command -volume-math '(1 - GM) * (sd > 0)'" 
            + " %s -var GM %s -var sd %s" % (csf, gm, sgndist))
        subprocess.run(cmd, shell=True)

    # Merge hemis 
    load = lambda f: nibabel.load(f).get_fdata().astype(np.float32).flatten()

    LPVs = np.stack(tuple(load(img) for img in (lgm,lwm,lcsf)), axis=1)
    RPVs = np.stack(tuple(load(img) for img in (rgm,rwm,rcsf)), axis=1)
    PVs = np.zeros_like(LPVs)

    PVs[:,0] = np.minimum(1, LPVs[:,0] + RPVs[:,0])
    PVs[:,1] = np.minimum(1 - PVs[:,0], LPVs[:,1] + RPVs[:,1])
    PVs[:,2] = 1 - np.sum(PVs[:,0:2], axis=1)

    assert np.all(PVs < 1 + 1e-6), 'PV higher than 1'
    assert np.all(np.sum(PVs, axis=1) < 1 + 1e-6), 'PVs sum to higher than 1'
    PVs = np.minimum(PVs, 1)

    # Save output 
    spc = t.classes.ImageSpace(ref)
    spc.saveImage(PVs.reshape((*spc.imgSize, 3)), outname)


def make_rot_matrix(deg, shift):

    a,b,c = (deg * (np.random.rand(3) - 0.5)) * np.pi/180
    trans = shift * (np.random.rand(3) - 0.5)

    rX = rotX(a)
    rY = rotY(b)
    rZ = rotZ(c)
    T = np.eye(4)
    T[0:3,3] = trans
    return T @ (rZ @ (rY @ rX))


def sum_within_space(img, ref): 


    if type(ref) is str: 
        refSpc = ImageSpace(ref)
    else: 
        assert isinstance(ref, ImageSpace)

    input = nibabel.load(img)
    inSpc = ImageSpace(img)

    factor = refSpc.vox_size / inSpc.vox_size
    if np.any(np.mod(factor, np.ones_like(factor))): 
        raise RuntimeError("Ratio of reference to input voxel size must be an integer")

    if np.any(factor != factor[0]):
        raise RuntimeError("Rescaling factor must be constant in all dimensions")

    inshp = input.shape 

    # Create zeros, sized according to the FoV of reference with voxels sized
    # according to the reference. Then write in the input data 
    outshp = tuple( (d * refSpc.vox_size[idx] / inSpc.vox_size[idx]).astype(np.int16) for idx,d in enumerate(refSpc.imgSize) )
    data = input.get_fdata()

    if len(data.shape) == 3: 
        frames = 1 
    elif len(data.shape) == 4: 
        frames = data.shape[3]
    else: 
        raise RuntimeError("Data size greater than 4D")

    output = np.zeros((*outshp, frames), dtype=np.float32)
    output[:inshp[0], :inshp[1], :inshp[2], :] = data[:,:,:,...]

    # Now sum across blocks for output, and don't forget to average out!
    summedout = t.resampling._sumArrayBlocks(output, (*factor, 1)) / (factor[0] ** 3)

    return np.squeeze(summedout)


def crop_image(img, out, start_extents, end_extents):
    """
    Crop image to within start_extends / end_extents (both must
    be 3-vectors of voxel numbers). img and out must both be paths 
    """

    spc = ImageSpace(img)
    crop_space = spc.crop(start_extents, end_extents)
    data = nibabel.load(img).get_fdata()
    s = start_extents; e = end_extents
    crop_space.save_image(data[s[0]:e[0], s[1]:e[1], s[2]:e[2], ...], out)


def downsample_image(img, out, factor, mode='floor'):
    """
    Downsample image by summation. 

    Args: 
        img: path to input
        out: path to save output 
        factor: 3-vector, neighbourhoods of voxels sized AxBxC will be summed 
        mode: 'floor' [default] or 'ceil'. Round the size of the subspace up
            or down if the FoV of the target size does not match the input 
    """

    if len(factor) != 3: 
        raise RuntimeError("Factor must be a list of 3 ints")

    spc = ImageSpace(img)
    down_space = spc.subsample(factor, mode)
    data = nibabel.load(img).get_fdata()
    lims = (down_space.size * factor).astype(np.int16)
    if len(data.shape) > 3: 
        factor = (*factor, 1)
    down_data = t.resampling._sumArrayBlocks(data[:lims[0], :lims[1], :lims[2], ...], factor)
    down_space.save_image(down_data, out)

    
def rotX(a):
    mat = np.eye(4)
    mat[(1,2), (1,2)] = np.cos(a)
    mat[1,2] = -np.sin(a)
    mat[2,1] = np.sin(a)
    return mat

def rotY(a):
    mat = np.eye(4)
    mat[(0,2), (0,2)] = np.cos(a)
    mat[0,2] = np.sin(a)
    mat[2,0] = -np.sin(a)
    return mat 

def rotZ(a):
    mat = np.eye(4)
    mat[(0,1), (0,1)] = np.cos(a)
    mat[0,1] = -np.sin(a)
    mat[1,0] = np.sin(a)
    return mat 

def resample(src, dest, ref, aff=None, flirt=False):

    od = op.dirname(dest)

    # Write out an I matrix into the folder for use. 
    if aff is None: 
        aff = np.eye(4)
    
    if type(aff) is str: 
        aff = np.loadtxt(aff)

    if not flirt:
        aff = t.utils._world_to_FLIRT(src, ref, aff)

    flirtmat = op.join(od, 'flirt_%s.txt' % hash(dest))
    np.savetxt(flirtmat, aff)

    cmd = ('applywarp -i %s -r %s -o %s --premat=%s --super --superlevel=a'
        % (src, ref, dest, flirtmat))
    subprocess.run(cmd, shell=True)

    subprocess.run('rm %s' % (flirtmat), shell=True)

def masked_vox_diff(ref,img,mask=None):
    """
    Mean absolute voxel difference between images ref and img
    If no mask is given, the following will be used:
    - for GM voxels, (GM in ref)
    - for WM voxels, (WM in ref)
    """

    # func = lambda a,b: np.mean()

    if mask is None: 
        mask = (ref[:,0:2] > 0)
        return 100 * np.sqrt([
            np.mean(((ref-img)[mask[:,0],0])**2, axis=0), 
            np.mean(((ref-img)[mask[:,1],1])**2, axis=0)
        ])

    else:
        return 100 * np.sqrt(np.mean(
            ((ref-img)[mask,0:2])**2, axis=0))