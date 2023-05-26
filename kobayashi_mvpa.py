# %%
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

data_dir_path = 'C:/Users/Maximilien/OneDrive - UGent/Case_studies_analysis_of_exp_data/data/' #change this to your data directory

# Make a nested list to refer to different participant paths to func files indexed like this [subject,run]
sub_dir_path = glob.glob(data_dir_path + 'sub-*')
paths = []

for ind,name in enumerate(sub_dir_path):        # for the func files
    func_fnames = glob.glob(name + '/func/*.nii.gz')
    paths.append(func_fnames)




# %%
"""

"""




# %%
plotting.view_img(image.mean_img(fname), threshold=None)


