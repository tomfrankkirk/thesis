#%%
"""
Draw graphs for the Toblerone paper from the simulated cortical surface
dataset
"""

import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 
savekws = {'dpi': 400, 'bbox_inches': 'tight'}
cmap = np.array(plt.get_cmap('tab10').colors)


#%% Load in the data, set some handy constants 

data = sio.loadmat('sim_surface_data.mat')
VOXSTEPS = np.arange(1, 3.2, 0.2)
TISSUES = ['GM', 'WM']
resamp = data['resamp']
voxs = data['voxs']
sums = data['sums']
all_methods = ['Tob', 'RC', 'RC2', 'Neuro', 'Neuro2']
start_idx = [0, 0, 1, 0, 1, 1]

#%% Error in total tissue volume at each resolution
# We use the numerical 1mm result as the ground truth, to which we compare all 
# other methods inclduing the numerical sln at other resolutions
truth = sums[0,-1,:]
sum_errs = 100 * (sums[:,:,:] - truth[None,None,:]) / truth[None,None,:]

for exclude,lims,suff in zip([['Neuro'], []], [(-0.08, 0.1), (-3, 1.5)], ['', '_full']):
    fig = plt.figure()
    fig.set_size_inches(7,3.5)
    axes = [ plt.subplot(1,2,i) for i in range(1,3) ]
    min_max = np.zeros((0,2))

    for tiss,ax in enumerate(axes):
        for midx,meth in enumerate(all_methods): 
            if meth not in exclude: 
                to_plot = sum_errs[start_idx[midx]:,midx,tiss]
                min_max = np.vstack((min_max, [to_plot.min(), to_plot.max()]))
                ax.plot(VOXSTEPS[start_idx[midx]:], to_plot, label=meth, color=cmap[midx,:])

        ax.plot(VOXSTEPS[1:], sum_errs[1:,-1,tiss], label='Num. sln.', color=cmap[5,:])
        ax.set_title('Error in total %s volume' % TISSUES[tiss])
        ax.set_xticks(VOXSTEPS)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('Voxel size (mm)')
        ax.set_ylim(lims)

    axes[0].legend(ncol=2)
    axes[0].set_ylabel('Error (%)')
    plt.savefig('figs/sim_sum%s.png' % suff, **savekws)


#%% Voxel-wise error at each resolution 
# We use the numerical solution at each voxel size as reference 

for exclude,lims,suff in zip([['Neuro'], []], [15, 21], ['', '_full']):
    fig = plt.figure()
    fig.set_size_inches(7,3.5)
    axes = [ plt.subplot(1,2,i) for i in range(1,3) ]
    min_max = np.zeros((0,2))

    for tiss,ax in enumerate(axes):
        for midx,meth in enumerate(all_methods): 
            if meth not in exclude: 
                to_plot = voxs[start_idx[midx]:,midx,tiss]
                min_max = np.vstack((min_max, [to_plot.min(), to_plot.max()]))
                ax.plot(VOXSTEPS[start_idx[midx]:], to_plot, label=meth, color=cmap[midx,:])

        ax.set_title('Per-voxel error in %s' % TISSUES[tiss])
        ax.set_xticks(VOXSTEPS)
        ax.tick_params(axis='x', labelrotation=90)        
        ax.set_xlabel('Voxel size (mm)')
        ax.set_ylim(0,lims)

    axes[1].legend(ncol=2)
    axes[0].set_ylabel('RMS voxel error (%)')
    plt.savefig('figs/sim_rms%s.png' % suff, **savekws)


#%% Error of resampling ground truth to other resolutions 

fig = plt.figure()
fig.set_size_inches(6,4)
axes = plt.subplot(1,1,1)
axes.hold = True 


for vidx in range(resamp.shape[1] - 2):
    axes.plot(VOXSTEPS[vidx+1:], resamp[vidx,vidx+1:,0], 
    label=('%1.1f' % VOXSTEPS[vidx]))

axes.legend(loc='center left', bbox_to_anchor=(1, 0.5),
    fancybox=True, ncol=1, title='Input voxel \n size (mm)')
plt.xticks(VOXSTEPS[1:])
plt.xlim(1.1, 3.1)
plt.xlabel('Output voxel size (mm)')
plt.ylabel('RMS voxel error (%)')
plt.title('Error of resampling ground truth (GM)')
plt.savefig('figs/sim_resamp.png', **savekws)
