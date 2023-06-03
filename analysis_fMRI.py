"""
Sam Vandermeulen 

"""
# links: 
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://www.sciencedirect.com/science/article/pii/S1053811921004225    
# https://nilearn.github.io/stable/decoding/decoding_intro.html
# https://nilearn.github.io/stable/manipulating_images/input_output.html#loading-data
# https://openneuro.org/datasets/ds003758/versions/1.0.2
# https://gitlab.com/kenji.k/beadsVOI
# https://neurovault.org/images/65086/
# https://nilearn.github.io/dev/modules/generated/nilearn.masking.apply_mask.html
# https://nilearn.github.io/dev/manipulating_images/masker_objects.html

###############################################################################

# import 
import nilearn 
import nibabel as nib
from nilearn import plotting, image, datasets 
from nilearn import image as nli
from nilearn.image import smooth_img, mean_img, index_img, resample_to_img
from nilearn.plotting import plot_epi, show, plot_roi
from nilearn.masking import compute_epi_mask, apply_mask
from nilearn.decoding import Decoder
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import numpy as n
from sklearn.svm import SVC
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import cross_val_score

###############################################################################

# load in fMRI data and couple behavioral labels per participant 
## participant 1 
result_p1 = nilearn.image.load_img(['D:/DATA_fMRI/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-1_desc-preproc_bold.nii.gz',
                                    'D:/DATA_fMRI/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-2_desc-preproc_bold.nii.gz',
                                    'D:/DATA_fMRI/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-3_desc-preproc_bold.nii.gz',
                                    'D:/DATA_fMRI/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-4_desc-preproc_bold.nii.gz'])

###############################################################################

# create condition mask: high and low reward
# behavioral labels in a .csv-file 
participant_1 = pd.read_csv('D:/DATA_fMRI/participant1/sub-BA3550/BA3550.csv')
# dict file 
choice = {1 : "high", 2 : "low"}
# replace 
participant_1.choice = [choice[item] for item in participant_1.choice]
print(participant_1.choice)
# extract values for boolean mask out of the column 
conditions = participant_1['choice'].values
# check the shape of the 4D fMRI image 
image_shape = result_p1.shape
# boolean mask based on conditions 
bool_mask = np.isin(range(image_shape[3]), np.where(np.isin(conditions, ["high", "low"]))[0])

###############################################################################

# create anatomical mask: left and right ventral striatum 
# import anatomical mask for bilateral ventral striatum (T1)
# Harvard-Oxford dubcortical atlas 
atlas_img = nilearn.datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-2mm", 
                                            data_dir = ('C:/Users/samvd/Desktop/CAES/DATA/anatomical mask'), 
                                            symmetric_split=False, resume=True, verbose=1)
# load atlas datasets and labels 
labels = atlas_img.labels
atlas_img = atlas_img.maps
# select regions out of the labels 
putamen_labels = ['Left Putamen', 'Right Putamen']
caudate_labels = ['Left Caudate', 'Right Caudate']
# make indices 
putamen_indices = [labels.index(label) for label in putamen_labels]
caudate_indices = [labels.index(label) for label in caudate_labels]
# Create a mask for the left and right putamen, and left and right caudate
putamen_mask = image.math_img('np.sum(np.stack([img == index for index in {}]), axis=0) > 0'.
                              format(putamen_indices), img = atlas_img)
caudate_mask = image.math_img('np.sum(np.stack([img == index for index in {}]), axis=0) > 0'.
                              format(caudate_indices), img = atlas_img)
# Visualize the left and right putamen masks
plotting.plot_roi(putamen_mask, title = 'Putamen Mask')
plotting.plot_roi(caudate_mask, title = 'Caudate Mask')
# Save the masks to NIfTI files
putamen_mask.to_filename('D:/DATA_fMRI/anatomical mask/putamen_mask.nii.gz')
caudate_mask.to_filename('D:/DATA_fMRI/anatomical mask/caudate_mask.nii.gz')
# merge putamen and caudate mask 
# Load the first mask
mask1_img = nilearn.image.load_img('D:/DATA_fMRI/anatomical mask/caudate_mask.nii.gz')
mask2_img = nilearn.image.load_img('D:/DATA_fMRI/anatomical mask/putamen_mask.nii.gz')
# Combine the two masks into a single mask image
combined_mask_img = image.math_img('mask1 + mask2', mask1 = mask1_img, mask2 = mask2_img)
# inspect mask 
plotting.plot_roi(combined_mask_img, title = "combined masks: bilateral ventral striatum")
# Resample the mask image to match the affine matrix of the fMRI image
# binary instead of linear or continuous, because the mask is binary
combined_mask_img_resampled = nilearn.image.resample_to_img(source_img = combined_mask_img, 
                                                            target_img = result_p1,
                                                            interpolation = 'nearest')                                                        

###############################################################################

# a first estimator (support vector classifier SVC with linear kernel)
## decoding (pipeline procedure to train model)
classifier = LinearSVC(max_iter=10000)
decoder = nilearn.decoding.Decoder(estimator = 'svc_l1',
                                   mask = combined_mask_img_resampled, 
                                   standardize = 'zscore_sample') 
decoder.fit(result_p1, bool_mask) 
print(f"SVC accuracy: {decoder.cv_scores.mean():.3f}")
#### K-fold strategy with decoder 
n_folds = 5
decoder = Decoder(estimator = "svc_l1", mask = combined_mask_img_resampled,
    standardize = 'zscore_sample', cv = n_folds, scoring = "accuracy",
)
decoder.fit(result_p1, bool_mask)
print(f"SVC accuracy: {decoder.cv_scores.mean():.3f}")

###############################################################################

# statistical test (k-fold cross-validation permutation test)
# define classifier 
svc = SVC(C = 1.0, kernel = "linear")
# only keep high and low
fmri_niimgs = index_img(result_p1, bool_mask)
# mask the data 
masker = NiftiMasker(mask_img = combined_mask_img_resampled,
                     smoothing_fwhm = 0, standardize="zscore_sample",
                     memory = "nilearn_cache", memory_level = 1)
# apply mask 
fmri_masked = masker.fit_transform(fmri_niimgs)
# Here `cv=5` stipulates a 5-fold cross-validation
cv_scores = cross_val_score(svc, fmri_masked, conditions, cv=5)
print(f"SVC accuracy: {cv_scores.mean():.3f}")

null_cv_scores = permutation_test_score(svc, fmri_masked, conditions, 
                                        cv = 5, n_permutations = 100, 
                                        groups = None)

print(f"permutation test score: {null_cv_scores[1].mean():.3f}")

## plot permutation null distribution and p-value 
fig, ax = plt.subplots()
ax.hist(null_cv_scores[1], bins=20, density=True)
ax.axvline(null_cv_scores[0], ls="--", color="r")

ax.set_xlabel(f"Accuracy score\naccuracy classifier: {null_cv_scores[0]:.2f}\n(p-value: {null_cv_scores[2]:.3f})")
_ = ax.set_ylabel("Probability")

###############################################################################

# rest code 
# Apply the combined mask to the fMRI bold data
masked_data = nilearn.masking.apply_mask(imgs = result_p1,
                                         mask_img = combined_mask_img_resampled, 
                                         dtype = 'f', smoothing_fwhm = None,
                                         ensure_finite = True)
# make the 2D data back 4D
masked_4Ddata = nilearn.masking.unmask(X = masked_data, 
                                       mask_img = combined_mask_img_resampled)

