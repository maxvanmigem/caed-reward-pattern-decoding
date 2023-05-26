# %%
"""
Load packages
"""

import nilearn, os, glob

from nilearn import plotting
from nilearn import image
from nilearn.decoding import Decoder

from sklearn.model_selection import KFold

import pandas as pd

# %%
"""
Create data paths and intialize base variables 
"""

data_dir_path = 'C:/Users/Maximilien/OneDrive - UGent/Case_studies_analysis-of_exp_data/data/ds003758/derivatives/' #change this to your data directory

# Make a nested lists to refer to different participant paths indexed like this [subject][run]
sub_dir_path = glob.glob(data_dir_path + 'sub-*')
anat_paths = []
fmap_paths = []
func_paths = []


for ind,name in enumerate(sub_dir_path):

    anat_fnames = glob.glob(name + '/anat/*.nii.gz')# for the anat files
    anat_paths.append(anat_fnames)

    fmap_fnames = glob.glob(name + '/fmap/*.nii.gz')# for the fmap files
    fmap_paths.append(fmap_fnames)

    func_fnames = glob.glob(name + '/func/*.nii.gz')# for the func files
    func_paths.append(func_fnames)


# the number of subjects can be derrived from the length of sub_dir_path list and the number
n_sub = len(sub_dir_path)
n_runs = len(anat_paths[0])


# %%
"""

"""

for i,e in enumerate(anat_paths[0]):
    print(e)



# %%
plotting.view_img(image.mean_img(func_paths[0][0]), threshold=None)

# %%
s1_r1 = nilearn.image.load_img(func_paths[0][0])

# %%
test_slice1 = image.index_img(func_paths[1][0],'right')
test_slice2 = image.index_img(func_paths[1][0],3)
plotting.view_img(test_slice1)

# %%
print(s1_r1.header)

# %%



