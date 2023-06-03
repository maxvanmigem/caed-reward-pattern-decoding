# %%
# Primary script for mvpa analysis of kobayashi et al. 2019 paper
# Written by Merel De Merlier, Sam Vandermeulen and Max Van Migem

# %%
# Import packages
import numpy as np
import pandas as pd
import nilearn,os,glob

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
bids_path = nilearn.interfaces.bids.get_bids_files('C:/Users/Carlos/ds003758/')

# the number of subjects can be derrived from the length of sub_dir_path list and the number
n_sub = len(sub_dir_path)
n_runs = len(func_paths[0])

TR = 1.5

# %%
""" 
Create behavioural labels from event files for model training i.e. label the fMRI timeslices
"""
# Make a very large dataframe with the behavioral labels
# Looped per subject
for sub in range(n_sub):
    # and looped per run
    for run in range(n_runs):   

        # read the behavior data  
        behavioral = pd.read_csv(event_paths[sub][run], sep= '\t')

        # create list with the length of the timeslices in this run
        # slice_dur = np.array([1.5 for x in nilearn.image.iter_img(func_paths[0][0])]) # safer alternative if runs dont have the same length
        slice_dur = np.array([1.5 for x in range(400)]) # faster

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
conditions = big_behav_set['bet_jar_type']
mask_rvs_fname = 'H:/Documents/cases_in_analysis_of_exp_data/R_VS.nii.gz'


# %%
fmri_roi_list = []
for sub in range(n_sub):

    for run in range(n_runs):
        downsampled_roi = nilearn.image.resample_to_img(source_img = mask_rvs_fname, target_img = func_paths[0][run], interpolation = 'nearest') 
        # Merge this all into a very large dataset
        fmri_roi_list.append(downsampled_roi)

fmri_roi_set = image.concat_imgs(fmri_roi_list,ensure_ndim=4)


# %%
fmri_roi_set = image.concat_imgs(fmri_roi_list,ensure_ndim=4)

# %%
import nibabel as nib

dimensions = fmri_roi_set.header.get_data_shape()
print(dimensions)

# %%
for i in range(len(anat_paths)):
    plotting.plot_roi(mask_rvs_fname, bg_img=anat_paths[i][0], cmap="Paired")

# %%
from sklearn.model_selection import KFold

cv = KFold(n_splits=5)

for fold, (train, test) in enumerate(cv.split(conditions), start=1):
    decoder = Decoder(
        estimator="svc", standardize="zscore_sample"
    )
    decoder.fit(image.index_img(fmri_roi_set, train), conditions[train])
    prediction = decoder.predict(image.index_img(fmri_roi_set, test))
    print(
        "CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(
            fold,
            (prediction == conditions[test]).sum()
            / float(len(conditions[test])),
        )
    )


