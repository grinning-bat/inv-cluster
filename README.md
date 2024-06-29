# Finding genomic regions with abberant distances pattern
This set of scripts takes as input: 
* A __.csv__ file, where first two columns are chromosome name and coordinate and the rest is a flattened matrix of pairwise distances (order doesn't matter, just keep it the same across lines).
* A __.bed__ mask file indicating regions accepted for further analysis
*  __chr_ends2.tsv__ - table with fields `chromosome_name length start end`
*  optionally: __lines.tsv__ to plot boundaries of genomic regions of interest

(fix file name patterns/csv format settings at the beginning of the scripts as needed)

Then `pwd_cluster.py` produces:
* A filtered __.tsv__ file (retained: rows having more than X sites in corresponding window marked as "good" in __.bed__ AND not outliers in PCA)
* A bunch of overview plots to check how efficient filtering was. Should be one or more nice ellipsoids without lonely stars and squised point clouds.

On the next step `pwd_all_cluster.py` and `barplots.py` take filtered __.tsv__ and produce:
* scatterplots along the chromosome with individual PC axes as Y coordinate and cluster ID as a color
* __.csv__ file with data for scatterplots
* manhattan-like plot for runs of windows with the same cluster ID in X (constant "`THR`" in`barplots.py`) of 10 neighbours to the left/right.
 