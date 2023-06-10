# %%
""" 
Script to extract, compute and plot roi images
"""
# Written by Merel De Merlier, Sam Vandermeulen and Max Van Migem

# %%
# Import packages
import numpy as np
import pandas as pd
import nilearn,os,glob

from nilearn import plotting
from nilearn import datasets
from nilearn import image
import nibabel as nib

# %%
""" 
Load and plot atlas images
"""
# Load atlas image of ventral striatum and medial orbitofrontal cortex 
# These atlas images are probabilistic so we binarize them first to get roi's with set boundaries
lvs_roi = image.binarize_img('H:/Documents/cases_in_analysis_of_exp_data/roi/FUP_FUNDUS_OF_PUTAMEN_VENTRAL_STRIATUM_LEFT.nii.gz', threshold=0.1)
rvs_roi = image.binarize_img('H:/Documents/cases_in_analysis_of_exp_data/roi/FUP_FUNDUS_OF_PUTAMEN_VENTRAL_STRIATUM_RIGHT.nii.gz', threshold=0.1)
lmofc_roi = image.binarize_img('H:/Documents/cases_in_analysis_of_exp_data/roi/AREA_FO3_OFC_LEFT.nii.gz', threshold=0.1)
rmofc_roi = image.binarize_img('H:/Documents/cases_in_analysis_of_exp_data/roi/AREA_FO3_OFC_RIGHT.nii.gz', threshold=0.1)

# Different combination to see if there are some that are more effective at predicting reward magnitude
bi_vs_roi = image.math_img('mask1 + mask2', mask1 = lvs_roi, mask2 = rvs_roi)
bi_mofc_roi = image.math_img('mask1 + mask2', mask1 = lmofc_roi, mask2 = rmofc_roi)
left_roi = image.math_img('mask1 + mask2', mask1 = lvs_roi, mask2 = lmofc_roi)
right_roi = image.math_img('mask1 + mask2', mask1 = rvs_roi, mask2 = rmofc_roi)
all_roi = image.math_img('mask1 + mask2 + mask3 + mask4', mask1 = lvs_roi, mask2 = lmofc_roi, mask3 = rvs_roi, mask4 = rmofc_roi)

# Plot these different images
plotting.plot_roi(lvs_roi,title='Left ventral striatum',black_bg=True)
plotting.plot_roi(rvs_roi,title='Right ventral striatum',black_bg=True)
plotting.plot_roi(lmofc_roi,title='Left mOFC',black_bg=True)
plotting.plot_roi(rmofc_roi,title='Right mOFC',black_bg=True)
plotting.plot_roi(bi_vs_roi,title='Bilateral ventral striatum',black_bg=True)
plotting.plot_roi(bi_mofc_roi,title='Bilateral mOFC',black_bg=True)
plotting.plot_roi(left_roi,title='Left hemisphere',black_bg=True)
plotting.plot_roi(right_roi,title='Right hemisphere',black_bg=True)
plotting.plot_roi(all_roi,title='Complete system',black_bg=True)

# %%
""" 
Save images
"""

nib.save(lvs_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/lvs_roi.nii.gz')
nib.save(rvs_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/rvs_roi.nii.gz')
nib.save(lmofc_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/lmofc_roi.nii.gz')
nib.save(rmofc_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/rmofc_roi.nii.gz')
nib.save(bi_vs_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/bi_vs_roi.nii.gz')
nib.save(bi_mofc_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/bi_mofc_roi.nii.gz')
nib.save(left_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/left_roi.nii.gz')
nib.save(right_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/right_roi.nii.gz')
nib.save(all_roi, 'H:/Documents/cases_in_analysis_of_exp_data/roi/clean_roi/all_roi.nii.gz')


