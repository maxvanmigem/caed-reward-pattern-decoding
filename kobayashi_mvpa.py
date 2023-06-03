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
behavioral = pd.read_csv(event_paths[0][0], sep= '\t')
# slice_dur = np.array([1.5 for x in nilearn.image.iter_img(func_paths[0][0])]) # safer
slice_dur = np.array([1.5 for x in range(400)]) # faster
slice_onset = np.zeros(len(slice_dur))

for ind,dur in enumerate(slice_dur):
    if ind == 0:
        slice_onset[ind] = TR
    else:
        slice_onset[ind] = slice_onset[ind-1] + TR

slice_timing = {'slice_onset': slice_onset, 'duration' : slice_dur}
pre_label_df = pd.DataFrame(slice_timing)


# %%
sub_trial_set = behavioral.loc[behavioral['event_type']=='bet']
onset_ndarr = sub_trial_set['onset'].to_numpy()
trial_ndarr = np.zeros(400)

count = 0
trt = 0
while count< len(slice_onset)-1:
    if slice_onset[count] < onset_ndarr[trt]:
        trial_ndarr[count+1] = trt
        count += 1
    else:
        trt +=1

pre_label_df['trial'] = trial_ndarr
subset_behav = sub_trial_set[['onset','points_high','bet_jar_type']].reset_index()
label_df = pd.merge(pre_label_df, subset_behav,how= 'left', left_on=['trial'],right_index= True)

# %%
conditions = label_df['bet_jar_type']

new_behav = pd.read_csv('H:/Documents/cases_in_analysis_of_exp_data/behavdata\BA3550.csv')
mask_rvs_fname = 'H:/Documents/cases_in_analysis_of_exp_data/R_VS.nii.gz'


# %%
for i in range(len(anat_paths)):
    plotting.plot_roi(mask_rvs_fname, bg_img=anat_paths[i][0], cmap="Paired")

# %%
fmri_niimgs_train = image.index_img(func_paths[0][0], slice(0, -30))
fmri_niimgs_test = image.index_img(func_paths[0][0], slice(-30, None))
conditions_train = conditions[:-30]
conditions_test = conditions[-30:]

decoder = Decoder(
    estimator="svc", mask=mask_rvs_fname, standardize="zscore_sample"
)
decoder.fit(fmri_niimgs_train, conditions_train)

prediction = decoder.predict(fmri_niimgs_test)

# The prediction accuracy is calculated on the test data: this is the accuracy
# of our model on examples it hasn't seen to examine how well the model perform
# in general.

print(
    "Prediction Accuracy: {:.3f}".format(
        (prediction == conditions_test).sum() / float(len(conditions_test))
    )
)


