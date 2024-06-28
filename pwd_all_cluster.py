#!/usr/bin/python3
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from itertools import combinations
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm

jn = os.path.join
data_dir="/home/user/CICHLID/"
make_scatters = False

# we will take list of chromosomes from here
chr_ends=pd.read_csv(jn(data_dir, "chr_ends2.tsv"),sep="\t")

# read the first filtered tsv file of distances
chrom=chr_ends['chr'][0]
c=pd.read_csv(jn(data_dir, f"filtered_{chrom}.tsv"), sep='\t',index_col=[0,1,2],header=[0,1])
# then concatenate others to it
for chrom in chr_ends['chr'][1:]:
  c=c.append(pd.read_csv(jn(data_dir, f"filtered_{chrom}.tsv"), sep='\t',index_col=[0,1,2],header=[0,1]))

# extract genomic coordinates
coords=[i[0] for i in c.index.values]

# extract chromosome number (as integer) to color dots later
# will fail if chromosome names are not in format '.{2}[0-9]+'
chrom_n=[int(i[2][3:]) for i in c.index.values]

#reduced_data = PCA(n_components=100).fit_transform(c)

# this is filtration, scaling and PCA
# WARNING: work in progress
# I showed at meeting just PCA(n_comp.=10)->StandardScaler
# see scklearn docs for what different scalers do

# PC's to retain
n_pca=100
#pca=PCA(n_components=100)
#pca.fit(c)
#reduced_data = pca.transform(c)

scaler = Normalizer()
scaler.fit(c)
normed_c = scaler.transform(c)

pca=PCA(n_components=n_pca)
pca.fit(normed_c)
reduced_data = pca.transform(normed_c)

scaler = StandardScaler()
scaler.fit(reduced_data)
reduced_data = scaler.transform(reduced_data)

DF_index_values=pd.DataFrame([list(i) for i in c.index.values])
DF_index_values.columns=['lin_coord','coord','chr']
DF_red_data=pd.DataFrame(reduced_data)
DF_red_data.columns=[f"PC{i}" for i in range(reduced_data.shape[1])]
DF_red_data.insert(0,"lin_coord",coords)
pd.merge(DF_index_values,DF_red_data,on='lin_coord',how='left').to_csv(f'all_chr_PCA{n_pca}.csv', sep=',',index=False)


if not make_scatters:
  exit

# scatter plot for all pairs of PC1,2,3,4 color by genomic coordinates
for i,j in combinations([0,1,2,3], 2):
  plt.figure(1)
  plt.clf()
  plt.scatter(reduced_data[:, i], reduced_data[:, j], marker=".", s=2,c=coords,cmap='viridis')
  plt.savefig(f"coords_all_chr_PC{i}_PC{j}_PCA{n_pca}.png")


# perform clustering
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=10, random_state=0)
kmeans.fit(reduced_data)
labels=kmeans.labels_

# You may try DBSCAN as well, but in needs a lot of eps tuning
#db = DBSCAN(eps=2.8, min_samples=10).fit(reduced_data)
#labels=db.labels_  

# scatter plot for all pairs of PC for which we suppose good
# inversion separation color by cluster
for i,j in combinations([1,2,9,6,8], 2):
  plt.figure(1)
  plt.clf()
  plt.scatter(reduced_data[:, i], reduced_data[:, j], marker=".", s=2,c=labels,cmap='Set1')
  plt.savefig(f"coords_all_chr_PC{i}_PC{j}_PCA{n_pca}.png")

for i in range(10):
  plt.figure(1)
  plt.clf()
  plt.scatter(coords, reduced_data[:, i], marker=".", s=2,c=labels,cmap='Set1')
  plt.savefig(f"kmeans_10_chr_all_coord_PC{i}_PCA{n_pca}.png")


# standard categorized palette has only 10 colors - make
# custom with that of chr number
hsv_cmap=plt.cm.get_cmap('hsv', 25)
chr_numbers=list(set(chrom_n))
np.random.shuffle(chr_numbers)
custom_cmap = ListedColormap(hsv_cmap(chr_numbers))

# scatter plot for all pairs of PC1,2,3,4 color by chromosome number
for i,j in combinations([0,1,2,3], 2):
  plt.figure(1)
  plt.clf()
  plt.scatter(reduced_data[:, i], reduced_data[:, j], marker=".", s=2,c=chrom_n,cmap=custom_cmap)
  plt.savefig(f"chr_all_chr_PC{i}_PC{j}_PCA{n_pca}.png")


# make plots for PC3,2,1/"genomic coordinate", with colors by chromosome 
plt.figure(1)
plt.clf()
plt.scatter(y=reduced_data[:, 2], x=coords, marker=".", s=2,c=chrom_n,cmap=custom_cmap)
plt.savefig(f"chr_all_chr_PC2_coord_PCA{n_pca}.png")

plt.figure(1)
plt.clf()
plt.scatter(y=reduced_data[:, 1], x=coords, marker=".", s=2,c=chrom_n,cmap=custom_cmap)
plt.savefig(f"chr_all_chr_PC1_coord_PCA{n_pca}.png")

plt.figure(1)
plt.clf()
plt.scatter(y=reduced_data[:, 0], x=coords, marker=".", s=2,c=chrom_n,cmap=custom_cmap)
plt.savefig(f"chr_all_chr_PC0_coord_PCA{n_pca}.png")

# get min and max values for PC3 - wee need this to draw chr. boundaries
(min_y,max_y)=reduced_data[:, 2].min(),reduced_data[:, 2].max()

# Make plots for different number of clusters
out_PC=1
for cl_n in range(2,10):
  kmeans = KMeans(init="k-means++", n_clusters=cl_n, n_init=10, random_state=0)
  kmeans.fit(reduced_data)
  labels=kmeans.labels_
  # bigger scatter plot PC3/"genomic coordinate" color by cluster
  plt.figure(1)
  plt.clf()
  plt.scatter(y=reduced_data[:, out_PC], x=coords, marker=".", s=2,c=labels,cmap='Set1')
  prev_x=0
  # plot chromosomes boundaries
  for ch_end in chr_ends.iloc:
    xi=ch_end['end']
    plt.axline((xi, min_y), (xi, max_y),lw=0.5,color="gray")
    plt.text(x=(xi+prev_x)/2,y=min_y+(min_y+max_y)*0.2/2,s=ch_end['chr'][3:],fontsize='small',ha='center')
    prev_x=xi
  plt.savefig(f"kmeans_{cl_n}_all_chr_PC{out_PC}_PCA{n_pca}_coord.png",dpi=300)