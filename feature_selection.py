# %%
# Reward decoding analysis kobayashi et al. 2021 dataset
# Written by Merel De Merlier, Sam Vandermeulen and Max Van Migem

# %%
# Import packages
import numpy as np
import pandas as pd
import nilearn,os,glob
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn import image
from nilearn import maskers
from nilearn.decoding import Decoder
from sklearn.model_selection import KFold


# %%
"""
Create data paths and intialize base variables 
"""

# data_dir_path = 'C:/Users/Maximilien/OneDrive - UGent/Case_studies_analysis-of_exp_data/data/ds003758/derivatives/' #change this to your data directory
data_dir_path = 'C:/Users/Carlos/ds003758/' # change this to wher your data is stored

# Make a nested lists to refer to different participant derivative paths indexed like this -> [subject][run]
sub_dir_path = glob.glob(data_dir_path +'derivatives/fsl/sub-*' )
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

# Do the same for event files
raw_dir_path = glob.glob(data_dir_path +'sub-*' )
event_paths = []

for ind,name in enumerate(raw_dir_path):
    event_fnames = glob.glob(name + '/func/*_events.tsv')# for the func files
    event_paths.append(event_fnames)



# Here is a different method but this just gives a list with all paths for every file
bids_path = nilearn.interfaces.bids.get_bids_files(data_dir_path)

# Load the ROI's that we're goining to use


# the number of subjects can be derrived from the length of sub_dir_path list and the number
n_sub = len(sub_dir_path)
n_runs = len(func_paths[0])
sub_list_all = np.arange(n_sub)
# only subjects 1,2 and 11 had intact data across all 4 runs all other data in this set had at leats  one corrupted file 
# (p.s. had to exclude subject 11 anyway because the because the last run files had a incompatible affine with the subjects other files)
sub_list = sub_list_all[np.array([obs_i in [0,1] for obs_i in sub_list_all])]

TR = 1.5

# %%
""" 
Create behavioural labels from event files for model training i.e. label the fMRI timeslices
"""
# Make a very large dataframe with the behavioral labels
# Looped per subject
for sub in sub_list:
    # and looped per run
    for run in range(n_runs):   

        # read the behavior data  
        behavioral = pd.read_csv(event_paths[sub][run], sep= '\t')

        # create list with the length of the timeslices in this run
        # slice_dur = np.array([1.5 for x in nilearn.image.iter_img(func_paths[0][0])]) # safer alternative if runs dont have the same length
        slice_dur = np.array([TR for x in range(400)]) # faster

        # make another list to idicating the timing of the slices
        slice_onset = np.zeros(len(slice_dur))

        for ind,dur in enumerate(slice_dur):
            if ind == 0:
                slice_onset[ind] = TR
            else:
                slice_onset[ind] = slice_onset[ind-1] + TR

        # combine these lists in a dataframe
        slice_timing = {'slice_onset': slice_onset, 'duration' : slice_dur}
        pre_label_df = pd.DataFrame(slice_timing)

        # 'bet' represents the end of a trial in the event files and thes rows also contain al the trial info we want, so we only select these  
        sub_trial_set = behavioral.loc[behavioral['event_type']=='bet']

        # we now use this subset to label the fmri slices to the apropriate trials  
        onset_ndarr = sub_trial_set['onset'].to_numpy()
        trial_ndarr = np.zeros(400)
        # to do this we check for each slice when they occur in relation to the behavioral onset times
        count = 0
        trt = 0
        while count< len(slice_onset)-1:
            if trt >= len(onset_ndarr):
                trial_ndarr[count+1] = 999
                trt +=1
                count +=1
            elif slice_onset[count] < onset_ndarr[trt]:
                trial_ndarr[count+1] = trt
                count += 1
            else:
                trt +=1

        pre_label_df['trial'] = trial_ndarr

        # Combine these arrays into a nice pandas dataframe
        subset_behav = sub_trial_set[['onset','points_high','bet_jar_type']].reset_index()
        label_df = pd.merge(pre_label_df, subset_behav,how= 'left', left_on=['trial'],right_index= True)

        # Add columns indicating to which subject and  to which run this data belongs 
        label_df['subject'] = sub
        label_df['run'] = run

        # Merge this all into a very large dataset
        if (sub == 0) and (run == 0):
            big_behav_set = label_df
        else: 
            big_behav_set = pd.concat([big_behav_set,label_df],axis=0, ignore_index=True)


# %%
"""
Aggregate nifti's per trial and apply mask
"""
# create paths to roi files, this is the ony thing that needs to change if we want to compare different roi's
l_vs_fname = './roi/clean_roi/lvs_roi.nii.gz'
r_vs_fname = './roi/clean_roi/rvs_roi.nii.gz'

l_mofc_fname = './roi/clean_roi/lmofc_roi.nii.gz'
r_mofc_fname = './roi/clean_roi/rmofc_roi.nii.gz'

bi_vs_fname = './roi/clean_roi/bi_vs_roi.nii.gz'
# bi_mofc_fname = './roi/clean_roi/bi_mofc_roi.nii.gz'

# all_fname = './roi/clean_roi/all_roi.nii.gz'

# Loop the selected subject list
for sub in sub_list:
    # Select this subjects behavioral data (some run had more behavioral data than fMRI data so we exlude trial 999)
    sub_data = big_behav_set.loc[(big_behav_set['subject']==sub) & (big_behav_set['trial'] != 999)].reset_index(drop=True)
    nifti_sublist = []

    # Loop for each run
    for run in range(n_runs):
        # Select this runs data with index corresponding to the time slices
        run_data = sub_data.loc[sub_data['run']==run].reset_index(drop=True)
        # List of trials with index corresponding to the time slices
        trial_chunks = run_data['trial']
        # Load the functional file
        run_niimg = image.load_img(func_paths[sub][run])
        # Iterate across each trial and take the mean of the slices that where captured on that trial
        for tr in np.unique(trial_chunks):
            trial_mask = trial_chunks.isin([tr])
            trial_mean = image.mean_img(image.index_img(run_niimg, trial_mask))
            nifti_sublist.append(trial_mean)

    # Concatinate these means together 
    subject_img = image.concat_imgs(nifti_sublist)
    # Select the apropriate labels for these mean images
    sub_trial_data = sub_data.drop_duplicates(subset=['trial','subject','run'] ,keep='last')
    # Keep the run info so that this is acounted for in the masking below
    session = sub_trial_data['run']

    # Intialize the mask that applies roi, normalizes and transform the data to 2D time series for each region
    # For the bilateral ventral striatium
    l_vs_masker = maskers.NiftiMasker(
        mask_img=l_vs_fname,
        standardize="zscore_sample",
        runs=session,
        smoothing_fwhm=4,
        memory="nilearn_cache",
        memory_level=1,
    )
    # For the bilateral medial Orbitofrontal cortex
    r_vs_masker = maskers.NiftiMasker(
        mask_img=r_vs_fname,
        standardize="zscore_sample",
        runs=session,
        smoothing_fwhm=4,
        memory="nilearn_cache",
        memory_level=1,
    )
    # For the for both combined
    l_mofc_masker = maskers.NiftiMasker(
        mask_img=l_mofc_fname,
        standardize="zscore_sample",
        runs=session,
        smoothing_fwhm=4,
        memory="nilearn_cache",
        memory_level=1,
    )
    # For the for both combined
    r_mofc_masker = maskers.NiftiMasker(
        mask_img=r_mofc_fname,
        standardize="zscore_sample",
        runs=session,
        smoothing_fwhm=4,
        memory="nilearn_cache",
        memory_level=1,
    )



    # Apply masks
    x_lvs_sub = l_vs_masker.fit_transform(subject_img)
    x_rvs_sub = r_vs_masker.fit_transform(subject_img)
    x_lmofc_sub = l_mofc_masker.fit_transform(subject_img)
    x_rmofc_sub = r_mofc_masker.fit_transform(subject_img)
    # x_all_sub = all_masker.fit_transform(subject_img)

    # Store these in a bigger set
    sub_labels = sub_trial_data['bet_jar_type']
    if (sub == 0):
        x_lvs = x_lvs_sub
        x_rvs = x_rvs_sub
        x_lmofc = x_lmofc_sub
        x_rmofc = x_rmofc_sub
        # x_all = x_all_sub
        
        labels = sub_labels
    else:
        x_lvs = np.vstack((x_lvs,x_lvs_sub))
        x_rvs = np.vstack((x_rvs,x_rvs_sub))
        x_lmofc = np.vstack((x_lmofc,x_lmofc_sub))
        x_rmofc = np.vstack((x_rmofc,x_rmofc_sub))
        # x_all = np.vstack((x_all,x_all_sub))

        labels = pd.concat([labels,sub_labels],axis=0, ignore_index=True)

# %%
""" 
Save the data
"""
np.save('./train_test_data/x_lvs.npy',x_lvs)
np.save('./train_test_data/x_rvs.npy',x_rvs)
np.save('./train_test_data/x_lmofc.npy',x_lmofc)
np.save('./train_test_data/x_rmofc.npy',x_rmofc)
labels.to_csv('./train_test_data/labels.csv')




