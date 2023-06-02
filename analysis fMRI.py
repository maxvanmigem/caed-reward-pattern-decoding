"""
Sam Vandermeulen 

"""
# links: 
# https://nilearn.github.io/stable/decoding/decoding_intro.html
# https://nilearn.github.io/stable/manipulating_images/input_output.html#loading-data
# https://openneuro.org/datasets/ds003758/versions/1.0.2
# https://gitlab.com/kenji.k/beadsVOI
# https://neurovault.org/images/65086/
# https://nilearn.github.io/dev/modules/generated/nilearn.masking.apply_mask.html
# https://nilearn.github.io/dev/manipulating_images/masker_objects.html

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
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import numpy as np

# load in fMRI data and couple behavioral labels per participant 
## participant 1 
result_p1 = nilearn.image.load_img(['C:/Users/samvd/Desktop/CAES/DATA/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-1_desc-preproc_bold.nii.gz',
                                     'C:/Users/samvd/Desktop/CAES/DATA/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-2_desc-preproc_bold.nii.gz',
                                     'C:/Users/samvd/Desktop/CAES/DATA/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-3_desc-preproc_bold.nii.gz',
                                     'C:/Users/samvd/Desktop/CAES/DATA/participant1/sub-BA3550/func/sub-BA3550_task-beads_run-4_desc-preproc_bold.nii.gz'])
# behavioral labels in a .csv-file 
participant_1 = pd.read_csv('C:/Users/samvd/Desktop/CAES/DATA/participant1/sub-BA3550/BA3550.csv')

# extract values for boolean mask out of the column 
conditions = participant_1['rew_corr_H'].values
# check the shape of the 4D fMRI image 
image_shape = result_p1.shape
# boolean mask based on conditions 
bool_mask = np.isin(range(image_shape[3]), np.where(np.isin(conditions, [170, 70, 7]))[0])
# apply the boolean mask on the 4D fMRI image
masked_fmri_img = nilearn.image.index_img(result_p1, bool_mask)
# print the shape of the boolean mask and the resulting masked fMRI image
print("Boolean mask shape:", bool_mask.shape)
print("Masked fMRI image shape:", masked_fmri_img.shape)

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
putamen_mask.to_filename('C:/Users/samvd/Desktop/CAES/DATA/anatomical mask/putamen_mask.nii.gz')
caudate_mask.to_filename('C:/Users/samvd/Desktop/CAES/DATA/anatomical mask/caudate_mask.nii.gz')

# merge putamen and caudate mask 
# Load the first mask
mask1_img = nilearn.image.load_img('C:/Users/samvd/Desktop/CAES/DATA/anatomical mask/putamen_mask.nii.gz')
mask2_img = nilearn.image.load_img('C:/Users/samvd/Desktop/CAES/DATA/anatomical mask/caudate_mask.nii.gz')

# Combine the two masks into a single mask image
combined_mask_img = image.math_img('mask1 + mask2', mask1 = mask1_img, mask2 = mask2_img)
# inspect mask 
plotting.plot_roi(combined_mask_img, title = "combined masks: bilateral ventral striatum")

# Resample the mask image to match the affine matrix of the fMRI image
# binary instead of linear or continuous, because the mask is binary
combined_mask_img_resampled = nilearn.image.resample_to_img(source_img = combined_mask_img, 
                                                            target_img = masked_fmri_img,
                                                            interpolation = 'nearest')

# Apply the combined mask to the fMRI bold data
masked_data = nilearn.masking.apply_mask(imgs = masked_fmri_img,
                                         mask_img = combined_mask_img_resampled, 
                                         dtype = 'f', smoothing_fwhm = None,
                                         ensure_finite = True)

# Print the shape of the masked data (246 time points and 2666 voxels)
print("Masked data shape:", masked_data.shape)

# a first estimator (support vector classifier SVC with linear kernel)
## decoding (pipeline procedure to train model)
classifier = LinearSVC(max_iter=10000)
decoder = Decoder(estimator = 'svc_l1',
                  mask = combined_mask_img_resampled, 
                  standardize = 'zscore_sample') 
decoder.fit(result_p1, bool_mask) 

## measuring prediction performance 
### cross-validation 
#### K-fold strategy -> e.g. k = 5
cv = KFold(n_splits=5)
for fold, (train, test) in enumerate(cv.split(bool_mask), start=1):
    decoder = Decoder(estimator="svc_l1", mask = combined_mask_img_resampled, standardize = 'zscore_sample'),
    decoder.fit(index_img(result_p1, train), bool_mask[train])
    prediction = decoder.predict(index_img(result_p1, test))
    print("CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(fold, 
                                                                (prediction == bool_mask[test]).sum()/ float(len(bool_mask[test])),))

#### K-fold strategy with decoder 
n_folds = 5
decoder = Decoder(estimator = "svc_l1", mask = combined_mask_img_resampled,
    standardize = 'zscore_sample', cv = n_folds, scoring = "accuracy",
)
decoder.fit(result_p1, bool_mask)

#### print accuracy 
print(decoder.cv_params_["7", "70", "170"])

### measure prediction accuracy (AUC for the ROC-curve)

# visualization 

# statistical test (permutation test)
permuted_ols()

