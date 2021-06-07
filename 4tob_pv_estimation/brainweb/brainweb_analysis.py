"""
Draw graphs for the Toblerone paper from the BrainWeb simulated
T1 dataset. The parameter space is sized (3 levels of non-uniformity) x
(6 levels of noise) x (4 voxel sizes)
"""
#%%
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
import toblerone
savekws = {'dpi': 400, 'bbox_inches': 'tight'}

#%% Load data from the matfile, define handy constants 

data = sio.loadmat('brainweb_data.mat')
VOXSIZES = np.arange(1,5)
NUS = np.array([0,20,40])
NOISES = np.array([0,1,3,5,7,9])
TISSUES = ['GM', 'WM']
sums = data['sums'] * (VOXSIZES ** 3)[None,None,:,None,None]
voxs = data['voxs']
structs = data['structs']
cmap = np.array(plt.get_cmap('Set1').colors)


#%% Difference in total tissue volume 
# The reference is NU = 0, Noise = 0, 1mm iso 

tref = sums[0,0,0,0,:]
fref = sums[0,0,0,1,:]
rcref = structs[0,0,0,1,-1]
scaler = 2e3
margin = 0.4

fig = plt.figure()
fig.set_size_inches(7,3.5)
axes = [ plt.subplot(1,2,i) for i in range(1,3,1) ]

for (tiss,ax),div in zip(enumerate(axes), [3,2]):
    ax.hold = True 

    for nuidx, nu in enumerate(NUS):
        tobpnts = scaler * np.abs(sums[nuidx,:,0,0,tiss] - tref[None,tiss]) / tref[None,tiss]
        fastpnts = scaler * np.abs(sums[nuidx,:,0,1,tiss] - fref[None,tiss]) / fref[None,tiss]
        dh1 = ax.scatter(2 * nuidx * np.ones(NOISES.shape) - margin, NOISES, tobpnts, c=cmap[1,None])
        dh2 = ax.scatter(2 * nuidx * np.ones(NOISES.shape) + margin, NOISES, fastpnts, c=cmap[0,None])

        if tiss == 0: 
            rcpnts = scaler * np.abs(structs[nuidx,:,0,1,-1] - rcref[None]) / rcref[None]
            dh3 = ax.scatter(2 * nuidx * np.ones(NOISES.shape), NOISES, rcpnts, c=cmap[2,None])
        
    ax.set_xticks([0,2,4])
    ax.set_xticklabels(NUS)
    ax.set_xlim([-1,5])
    ax.set_xlabel('NU (%)')
    ax.set_ylim([-1,10])
    ax.set_yticks(NOISES)
    ax.set_title('Difference in total %s volume' % TISSUES[tiss])

dhandles = [dh1, dh2, dh3]
dlabels = ['Tob', 'FAST', 'RC (cortex)']
axes[0].set_ylabel('Noise (%)')

dummydata = [0.01,0.05,0.15]
dummies = [ axes[0].scatter([0,1,2], [14,14,14], scaler * d, c=cmap[3,None]) for d in dummydata ]

labels = ['%d%%' % (100*i) for i in dummydata]
box = axes[0].get_position()
axes[0].legend(loc='lower center', bbox_to_anchor=(1.1, -0.32),
    fancybox=True, labels=labels+dlabels, ncol=6, handles=dummies+dhandles)

plt.savefig('figs/bw_sum.png', **savekws)



#%% RMS voxel difference 
# The reference is NU = 0, Noise = 0 @ each voxel size
markers = ["^", "D", "o", "s"]
xpos = [-1, 1, 1, -1]
ypos = [1, 1, -1, -1]
multi = 10
margin = 0.3
ymax = voxs.max()

fig = plt.figure()
fig.set_size_inches(7,12)
axes = [ fig.add_subplot(2,1,i) for i in range(1,3) ]

for tiss,ax in enumerate(axes): 
    for vidx, vox in enumerate(VOXSIZES):
        for nuidx, nu in enumerate(NUS):
            tobpnts =  multi * voxs[nuidx,:,vidx,0,tiss]
            fastpnts = multi * voxs[nuidx,:,vidx,1,tiss]
            xmarks = xpos[vidx] * (2 * nuidx * np.ones(NOISES.shape) + 1)
            ymarks = ypos[vidx] * NOISES + ypos[vidx]
            dh1 = ax.scatter(xmarks - margin, ymarks, tobpnts, cmap[1,None], markers[vidx])
            dh2 = ax.scatter(xmarks + margin, ymarks, fastpnts, cmap[0,None], markers[vidx])

    xticks = [ x+1 for x in range(-6,6,2) ]
    xlabels = [ '%d' % NUS[(abs(x)-1)//2] for x in xticks ]
    yticks = list(range(-10,-1,2)) + [-1,1] + list(range(2,11,2))
    ylabels = [ '%d' % n for n in NOISES[1:] ]
    ylabels = ylabels[::-1] + ['0','0'] + ylabels 

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim([-6,6])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.set_ylabel('Noise (%)')
    ax.set_title("RMS voxel difference in %s" % TISSUES[tiss])

ax.set_xlabel('NU (%)')

dhandles = [dh1, dh2]
dlabels = ['Tob', 'FAST']
dummysizes = np.array([1, 10, 15, 20])
handles = [ ax.scatter(-10, 10, multi * sz, cmap[2,None], sym)
    for sz,sym in zip(dummysizes, markers) ] 
labels = [ '%dmm voxels, %d%% diff.' % (v,e) for (v,e) in zip(VOXSIZES, dummysizes) ]

box = axes[0].get_position()
axes[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, labels=labels+dlabels, ncol=3, handles=handles+dhandles)

plt.savefig('figs/bw_voxs.png', **savekws)

#%% Subset of the above plot for 3mm results 

multi = 15
margin = 0.3
ymax = voxs[:,:,2,:,0].max()

fig = plt.figure()
fig.set_size_inches(7.5,3.5)
axes = [ fig.add_subplot(1,2,i) for i in range(1,3) ]

for tiss,ax in enumerate(axes): 
    for nuidx, nu in enumerate(NUS):
        tobpnts =  multi * voxs[nuidx,:,2,0,tiss]
        fastpnts = multi * voxs[nuidx,:,2,1,tiss]
        xmarks = 2 * (nuidx * np.ones(NOISES.shape) + 1)
        ymarks = (NOISES + ypos[vidx]) + 1
        dh1 = ax.scatter(xmarks - margin, ymarks, tobpnts, cmap[1,None])
        dh2 = ax.scatter(xmarks + margin, ymarks, fastpnts, cmap[0,None])

    ax.set_xticks([2,4,6])
    ax.set_xticklabels(['%d' % n for n in NUS])
    ax.set_title("RMS voxel difference, %s at 3mm" % TISSUES[tiss])
    ax.set_xlim(1,7)
    ax.set_xlabel('NU (%)')
    ax.set_ylim(-1,10)
    ax.set_yticks(NOISES)

axes[0].set_ylabel('Noise (%)')

dhandles = [dh1, dh2]
dlabels = ['Tob', 'FAST']
dummysizes = np.array([1, 5, 15])
handles = [ ax.scatter(-10, 10, multi * sz, cmap[2,None])
    for sz in dummysizes ] 
labels = [ '%d%%' % e for e in dummysizes ]


box = axes[0].get_position()
axes[1].legend(loc='lower center', bbox_to_anchor=(-0.1, -0.3),
          fancybox=True, labels=labels+dlabels, ncol=5, handles=handles+dhandles)

plt.savefig('figs/bw_voxs_subset.png', **savekws)


#%% RMS per-voxel differences, exploded

margin = 3
multi = 10 
fig = plt.figure()
fig.set_size_inches(7, 14)
axes = [] 

for v in range(4):
    for tiss in range(2):
        axes.append(plt.subplot(4,2, (v*2) + tiss + 1))
        plt.title('RMS voxel difference, %1dmm %s' % (VOXSIZES[v], TISSUES[tiss]))
        for nuidx,nu in enumerate(NUS):
            ymarks = NOISES
            xmarks =  nu * np.ones_like(NOISES)
            h1 = plt.scatter(xmarks - margin, ymarks, multi * voxs[nuidx,:,v,0,tiss], cmap[1,None])
            h2 = plt.scatter(xmarks + margin, ymarks, multi * voxs[nuidx,:,v,1,tiss], cmap[0,None])
            plt.xlim(-10,50)
            plt.yticks(NOISES, NOISES)
            plt.xticks([])

for x in [-1,-2]:
    axes[x].set_xlabel('NU (%)')
    axes[x].set_xticks(NUS)

for x in [0,2,4,6]:
    axes[x].set_ylabel('Noise (%)') 

dhandles = [h1, h2]
dlabels = ['Tob', 'FAST']
dummysizes = np.array([1, 10, 20])
handles = [ ax.scatter(-10, 10, multi * sz, cmap[2,None])
    for sz in dummysizes ] 
labels = [ '%d%%' % e for e in dummysizes ]


box = axes[0].get_position()
axes[6].legend(loc='lower center', bbox_to_anchor=(1.1, -0.4),
          fancybox=True, labels=labels+dlabels, ncol=5, handles=handles+dhandles)

plt.savefig('figs/bw_voxs_exploded.png', **savekws)


#%% Difference in individual FIRST + Cortex struct vol 
# Reference is NU = 0, Noise = 0, 1mm iso. 
# NOTE FAST IS NOT INCLUDED (we cannot mask FAST subcortical structures
# without biasing the results)

spacing = 5 
all_structs = toblerone.utils.STRUCTURES + ['Cortex', 'Cortex_RC']
multi = 1e3
xpos = lambda cent,off: (spacing*cent + (off-1)) * np.ones_like(NOISES)
ymax = structs.max()
dummy = [0.05, 0.15, 0.25]

fig = plt.figure()
fig.set_size_inches(6, 18)

# Plot the FIRST structs
for sidx in range(structs.shape[-1]):
    sref = structs[0,0,0,0,sidx]
    points = multi * np.abs(structs[:,:,0,0,sidx] - sref) / sref 
    for nuidx, nu in enumerate(NUS):
        plt.scatter(NOISES, xpos(sidx,nuidx), points[nuidx,:], c=cmap[nuidx,None])

# Add the cortex results
sref = structs[0,0,0,1,-1]
points = multi * np.abs(structs[:,:,0,1,-1] - sref) / sref 
dhandles = []
for nuidx, nu in enumerate(NUS):
        plt.scatter(NOISES, xpos(sidx+1,nuidx), points[nuidx,:], c=cmap[nuidx,None])
        dhandles.append(plt.scatter(NOISES, xpos(sidx+1,nuidx) + 10, 10 * points[nuidx,:], c=cmap[nuidx,None]))

plt.yticks(range(0, spacing*len(all_structs), spacing), all_structs)
plt.xlabel('Noise (%)')
plt.xticks(NOISES)
plt.ylabel('Structure')
plt.ylim([-2, spacing * (len(all_structs)) -3])
plt.xlim([-1,10])

handles = []
labels = []
for d in dummy: 
    handles.append(plt.scatter(20,20, multi * d, c=cmap[3,None]))
    labels.append('%d%% diff' % (100*d))
plt.legend(handles=handles+dhandles, 
    labels=labels+['%d%% NU' % n for n in NUS], bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3)
plt.title('Difference (%) in structure volume')

plt.savefig('figs/bw_all_structs.png', **savekws)


#%%
