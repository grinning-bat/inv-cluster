#!/usr/bin/python3
"""
This scripts get filtered tsv with pairwise distances from the previous stage and produces: 
  * manhattan-like plot for runs of windows with the same cluster ID in X 
    (constant "`THR`") of 10 neighbours to the left/right.
"""
import os
#from itertools import combinations
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
import numpy as np
#from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

WINDOW_SIZE = 100000
# will calculate and plot for N clusters [2..CL_MAX]
CL_MAX = 8
# will plot contigous runs of window with THR neighbors from the same cluster on the left or right
THR = 8
# scatter or barplot
PLOT_ALL = False
# PCA axes to retain
N_PCA = 100


jn = os.path.join
DATA_DIR = "/home/user/CICHLID/"
IND_FILE = "/home/user/up/GCA_900246225.3_fAstCal1.2_genomic_chromnames_mt.fa.fai"


#old_scale=cm.Set1.colors
old_scale = cm.tab20.colors
cc_list = old_scale + ((0,0,0),)
ccm = ListedColormap(cc_list)


# we will take list of chromosomes from here
genome_index = pd.read_csv( IND_FILE,
                            sep='\t',
                            header=None,
                            names="contig,contig_len,location,basesPerLine,bytesPerLine".split(',')
                          )
regular_chr = genome_index.contig.apply(lambda x: x in [f"chr{i}" for i in range(1,26)])

chr_ends=genome_index[regular_chr].copy()
chr_ends['end']=chr_ends.contig_len.cumsum()
chr_ends['start']=chr_ends.end-chr_ends.contig_len
chr_ends=chr_ends.rename(columns={'contig_len':'len','contig':'chr'})

# read the first filtered tsv file of distances
chrom=chr_ends['chr'][0]
c = pd.read_csv(
                 jn(DATA_DIR, f"filtered_{chrom}.tsv"),
                 sep='\t',index_col=[0,1,2],header=[0,1]
               )

# then concatenate others to it
for chrom in chr_ends['chr'][1:]:
    c = c.append( pd.read_csv(
                               jn( DATA_DIR, f"filtered_{chrom}.tsv"),
                               sep='\t', index_col=[0,1,2], header=[0,1]
                             )
                )

coord_df = pd.DataFrame( list(c.index),columns=['linear', 'in_contig', 'contig' ] )

coord_conv = coord_df.merge(chr_ends, left_on="contig", right_on="chr")
coord_conv['index_linear']=coord_conv['in_contig']+coord_conv['start']

coords=list(coord_conv['index_linear'])

scaler = Normalizer()
scaler.fit(c)
normed_c = scaler.transform(c)

pca=PCA(n_components = N_PCA)
pca.fit(normed_c)
reduced_data = pca.transform(normed_c)

scaler = StandardScaler()
scaler.fit(reduced_data)
reduced_data = scaler.transform(reduced_data)

plt.clf()
fig=plt.figure(1)
fig.set_size_inches(20,10)
gs = fig.add_gridspec(CL_MAX-1, hspace=0)
axs = gs.subplots(sharex=True)

# Perform KMeans with different N of clusters for each N save
# a list of neighbour counts with the same cluster ID choosing
# the max value from "neighbors to the left" and "... to the right"
# of every window
for cl_n in range(2,CL_MAX+1):
    kmeans = KMeans(init="k-means++", n_clusters=cl_n, n_init=50, random_state=0)
    kmeans.fit(reduced_data)
    labels=kmeans.labels_
    #
    out=[]
    for n,pos in enumerate(labels):
        l_count = 0
        r_count = 0
        this_cl = labels[pos]
        for i in range(1,10):
            if len(labels) > pos+i and labels[pos+i] == this_cl:
                r_count += 1
            else:
                break
        for i in range(1,10):
            if 0<pos-i and labels[pos-i]==this_cl:
                l_count += 1
            else:
                break
        out.append(max(l_count,r_count))
    #
    # fill the array of Y coordinates
    bars=np.zeros(max(coords)//WINDOW_SIZE+1,dtype=np.int8)
    bars[:]=ccm.N
    for cl,conf,pos in zip(labels,out,coords):
        if conf>THR:
            bars[pos//WINDOW_SIZE]=cl
    if CL_MAX>2:
        ax=axs[cl_n-2]
    else:
        ax=axs
    if PLOT_ALL:
        ax.scatter(y=out, x=coords, marker=".", s=1,c=labels,cmap=ccm,vmin=0,vmax=ccm.N)
        ax.set_ylim([0.5,11])
        ax.set_yticks([0,8])
    else:
        ax.imshow( bars[np.newaxis,:],
                   aspect = "auto",
                   cmap = ccm,
                   vmin = 0,
                   vmax = ccm.N,
                   interpolation ='none' )
        ax.set(yticklabels=[])
        ax.tick_params(left=False)
    ax.set_ylabel( f"K={cl_n}", fontsize = 'medium',
                   rotation = 'horizontal',
                   horizontalalignment = 'left',
                   labelpad = 50 )

# draw chromosome boundaries and labels
text_labels=[]
positions=[]
for ch_end in chr_ends.iloc:
    xe=ch_end['end']
    xs=ch_end['start']
    if PLOT_ALL:
        positions.append((xs+(xe-xs)/2))
        positions.append(xe)
        for ax in axs:
            ax.axvline(xe,c='black',lw=0.5)
    else:
        positions.append((xs+(xe-xs)/2)//WINDOW_SIZE)
        positions.append(xe//WINDOW_SIZE)
        for ax in axs:
            ax.axvline(xe//WINDOW_SIZE,c='white',lw=0.5)
    text_labels.append(ch_end['chr'])
    text_labels.append("")

plt.xticks(ticks=positions,labels=text_labels)
if PLOT_ALL:
    plt.savefig(f"skyline.cl_to{CL_MAX}.png",dpi=300)
else:
    plt.savefig(f"barplot.conf{THR}.cl_to{CL_MAX}.png",dpi=300)
