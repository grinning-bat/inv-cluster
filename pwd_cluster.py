#!/usr/bin/python3
# this script is intended to run under GNU parallel or PBS
# takes chromosome name as the first argument

import sys
# it is the modified version of pypopgen3 it should be installed locally
sys.path.append("/home/user/CICHLID/pypopgen3")
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
jn = os.path.join
from pypopgen3.modules import vcfpandas as vp
data_dir = '.'

# controls if we want to draw known inversion boundaries on plots
draw_lines=False
# set threshold for coverage (in n's of well called sites in window)
thr=80000

chrom = sys.argv[1]

# read pairwise distances from tsv file
# frist two columns - chromosome name and window coordinate
window_pwd = pd.read_csv(jn(data_dir, f"window_pwd.ops.{chrom}.100000.100000.diploid.tsv"), sep='\t',index_col=[0,1],header=[0,1])

# extract window start coordinates to separate list
# TODO: rewrite to use dataframe column as it is instead
coords=[i[1] for i in window_pwd.index.values]

# read coordinates at which chromosomes end
# will need this to make filtered files
# which we will join later to make plots for the whole genome 
chr_ends=pd.read_csv("chr_ends2.tsv",sep='\t')
chr_coord=chr_ends[chr_ends['chr']==chrom].iloc[0]
chr_start=chr_coord['start']

# read known inversion boundaries
if draw_lines:
  bound=pd.read_csv("lines.tsv",sep='\t')
  bound=bound[bound['chr']==chrom].iloc[0]
  x_lines=list(bound[['inv_start_left', 'inv_start_right', 'inv_end_left',  'inv_end_right']])

# filter by coverage from bed masks
accessible=[]
# that's a performance crime
# should be rewritten if we plan to take shorter windows
for i in coords:
  accessible.append(vp.get_accessible_size("./bed_masks/" + f"malawi_cichlids_callset_v3_qc_subset_{chrom}_pass.bed.gz", chrom, start=i-50000, end=i+50000))

# here is filtered set of df rows
# TODO add here column of coords, to avoid separate filtration of coords list
accessible_fixed=pd.DataFrame([row for row,cov in zip(window_pwd.iloc,accessible) if cov>thr])

# now filter coordinates list to match our DataFrame
coords=[i[1] for i in accessible_fixed.index.values]

# filter also accessible positions count, as we are going to 
# make a PCA plot with accessible count coloring
accessible_flt=list(filter(lambda x: x>thr, accessible))

# perform PCA
# maybe we should do additional outlier filtering here
# or take more robust dimensional reducing method
# I'm working on it now
reduced_data = PCA(n_components=10).fit_transform(accessible_fixed)

# scale data (optional)
scaler = StandardScaler()
scaler.fit(reduced_data)

reduced_data = scaler.transform(reduced_data)


# start making plots
# scatter "well covered sites"/"coordinate in chr"
plt.figure(1)
plt.clf()
plt.scatter(y=accessible_flt, x=coords, marker=".", s=2)
plt.savefig(f"accessible_by_len_{chrom}.png")

# scatter PC1/PC2 color by "coordinate in chr"
plt.figure(1)
plt.clf()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker=".", s=2,c=coords,cmap='viridis')
plt.savefig(f"coords_{chrom}.png")

# scatter PC1/PC2 color by "well covered sites"
plt.figure(1)
plt.clf()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker=".", s=2,c=accessible_flt,cmap='viridis')
plt.savefig(f"accessible_by_len_{chrom}.png")

# make clustering with outliers
# set 2 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, random_state=0)
kmeans.fit(reduced_data)
KMeans(n_clusters=2, random_state=0)
#kmeans.labels_
labels=kmeans.labels_

# scatter PC1/PC2 color by cluster
plt.figure(1)
plt.clf()
#labels = db.labels_
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker=".", s=2,c=labels,cmap='Set1')
plt.savefig(f"kmeans_scaled_{chrom}.png")
#plt.savefig(f"dbscan_scaled_{chrom}.png")

plt.figure(1)
plt.clf()
(min_y,max_y)=reduced_data[:, 0].min(),reduced_data[:, 0].max()

# scatter PC1/coordinate color by cluster
plt.scatter(y=reduced_data[:, 0], x=coords, marker=".", s=2,c=labels,cmap='Set1')
# plot known inversion boundaries
if draw_lines:
  for xi in x_lines:
    plt.axline((xi, min_y), (xi, max_y),lw=0.5,color="red")

plt.savefig(f"kmeans_scaled_{chrom}_coords.png")
# done!
##plt.savefig(f"dbscan_scaled_{chrom}_coords.png")

# outliers filtration
a=[np.where(np.abs(stats.zscore(column))>3) for column in reduced_data.T]
# we scanned through columns and marked every cell,
# which is outlier for its column
# if it was one - we returned its row number
# now we just collect these (repeating) numbers and
# take each only once
outliers=set(np.concatenate(a,axis=None))
# make array of zeroes of the same length as our data
outliers_color=[0]*len(reduced_data)
# go through the list with row numbers
# set 1 at every row with number from list
# in zero-filled array
for i in outliers:
  outliers_color[i]=1

# filter PCA-transformed data
reduced_data=reduced_data[(np.array(outliers_color)==0),:]
# perform clustering on outlier-free data
kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, random_state=0)
kmeans.fit(reduced_data)
labels=kmeans.labels_
(min_y,max_y)=reduced_data[:, 0].min(),reduced_data[:, 0].max()
#filter window coordinates list
coords=[n for c,n in enumerate(coords) if c not in outliers]


# scatter PC1/coords color by cluster
plt.figure(1)
plt.clf()
plt.scatter(y=reduced_data[:, 0], x=coords, marker=".", s=2,c=labels,cmap='Set1')
plt.savefig(f"kmeans_scaled_noout_{chrom}_coords.png")

# scatter PC1/PC2 color by cluster
plt.figure(1)
plt.clf()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker=".", s=2,c=labels,cmap='Set1')
plt.savefig(f"kmeans_scaled_noout_{chrom}.png")

# filter coverage list to match our _NOT_ PCA-transfromed and filtered data
accessible_fixed=accessible_fixed.iloc[(np.array(outliers_color)==0),:]
# add columns with genomic coordinates, chromosome coordinates and chromosome name
accessible_fixed.loc[:,('linear','coord')]=[i[1]+chr_start for i in accessible_fixed.index]
accessible_fixed.loc[:,('chr','coord')]=[i[1] for i in accessible_fixed.index]
accessible_fixed.loc[:,('chr','name')]=chrom
# put them at leftmost position
reorder_columns=accessible_fixed.columns[-3:]
reorder_columns=reorder_columns.append(accessible_fixed.columns[:-3])
# write filtered tsv to disk
accessible_fixed[reorder_columns].to_csv(f'filtered_{chrom}.tsv',index=False,sep="\t")