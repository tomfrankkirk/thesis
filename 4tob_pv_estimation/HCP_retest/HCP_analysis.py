#%% HCP analysis

import toblerone.utils 
import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np 

savekws = {'dpi': 400, 'bbox_inches': 'tight'}
cmap = np.array(plt.get_cmap('Set1').colors)

#%% Load data

data = sio.loadmat('HCP_data.mat')
sums = data['sums']
voxs = data['voxs']
vox_diffs = data['diffs']
structs = data['structs']
hist_bins = np.squeeze(data['hist_bins'])
VOXSIZES = np.hstack(([0.7], np.arange(1.0, 4.2, 0.4)))
TISSUES = ['GM', 'WM']
sums *= (VOXSIZES ** 3)[None,:,None,None,None]


#%% Difference in tissue volume from 0.7mm estimates between sessions 
# The reference is the session 1 data, to which we compare session 2

ref = sums[:,0,:,0,:]
sum_diffs = (100 * (sums[:,0,:,1,:] - ref) / ref)

fig = plt.figure()
fig.set_size_inches(7,4)
ax = fig.add_subplot(1,1,1)
ax.hold = True 

ax.plot([-2,8], [0,0], 'k-.', linewidth=0.5)
g_violins = ax.violinplot(sum_diffs[:,:,0], positions=range(0,7,3)) 
w_violins = ax.violinplot(sum_diffs[:,:-1,1], positions=range(1,6,3))

(c1,c2) = ( v['bodies'][0]._facecolors for v in (g_violins, w_violins) )
dh1 = ax.bar(8,1,color=c1)
dh2 = ax.bar(9,1,color=c2)

ax.set_xticks([0.5, 3.5, 6])
ax.set_xticklabels(['Toblerone', 'FAST', 'RC \n(cortex only)'])
ax.set_ylabel('Difference (%)')
ax.set_xlim([-1, 7])
ax.legend([dh1, dh2], ['GM', 'WM'], loc='lower right')
ax.set_title('Difference (retest - test) in tissue volume')

plt.savefig('figs/hcp_one_sum.png', **savekws)


#%% Difference between Tob and FAST PV estimates at each resolution 
# Sorted into 5% percentiles according to Toblerone's PV estimate 
# Not not all brains are the same size, and we want this test to be per-voxel
# So reweight the 5% bins using the subject/session brain volume - this 
# normalises out for brain volume so bigger brains with more voxels get more
# weight 
# sums is subs x voxels x method x sessions x tissue 
weights = sums[:,0,0,:,0].mean(-1)
weights = weights / weights.max()
flat_means = np.average(vox_diffs.mean(2), axis=0, weights=weights)
plot_bins = hist_bins[:-1] + 0.5*(hist_bins[1] - hist_bins[0])

for tiss in range(2):
    fig = plt.figure()
    fig.set_size_inches(7,5)

    for vidx,v in enumerate(VOXSIZES):
        plt.plot(plot_bins, flat_means[vidx,tiss,:], 
            label='%1.1fmm' % v, linewidth=1)
    
    plt.plot(plot_bins, np.zeros_like(plot_bins), 'k--', linewidth=1)    
    plt.xticks(plot_bins, labels=np.round(100*plot_bins,1), rotation=90)
    plt.legend(ncol=3)
    plt.xlabel('Toblerone GM PV (%)')
    plt.ylabel('Mean difference (Toblerone - FAST)')
    plt.title('Tob. - FAST %s PV difference, sorted by Tob. GM PV' % TISSUES[tiss])
    plt.xlim(0,1)

    plt.savefig('figs/hcp_vox_diffs_%s.png' % TISSUES[tiss], **savekws)


#%% Difference in struct vol wrt NU = 0, Noise = 0, 1mm iso. 
# Once again the same caveat as for the equivalent BrainWeb plot applies: 
# we cannot draw these for FAST as we cannot mask the results without biasing
# the analysis. 

all_structs = toblerone.utils.STRUCTURES + ['Cortex']
struct_ref = structs[:,0,:]
struct_diffs = 100 * (structs[:,1,:] - struct_ref) / struct_ref

fig = plt.figure()
fig.set_size_inches(12, 18)
zero_line = np.stack((np.zeros(len(all_structs)+2), np.arange(len(all_structs)+2)))
plt.plot(zero_line[0,:], zero_line[1,:], 'k-.', linewidth=0.5)
plt.violinplot(struct_diffs, vert=False)

plt.yticks(range(1,len(all_structs)+1), labels=all_structs)
plt.ylim(0.5, len(all_structs) + 0.5)
plt.title('Difference (retest - test) in structure volume')
plt.xlabel('Difference (%)')
plt.ylabel('Structure')

plt.savefig('figs/hcp_all_structs.png', **savekws)

#%%
