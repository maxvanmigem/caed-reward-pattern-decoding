# %%
from nilearn import datasets

# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
# 'func' is a list of filenames: one for each subject
fmri_filename = haxby_dataset.func[0]

# print basic information on the dataset
print(f"First subject functional nifti images (4D) are at: {fmri_filename}")

# %%
from nilearn import plotting
from nilearn.image import mean_img

plotting.view_img(mean_img(fmri_filename), threshold=None)

# %%
mask_filename = haxby_dataset.mask_vt[0]

# Let's visualize it, using the subject's anatomical image as a
# background
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0], cmap="Paired")

# %%
import pandas as pd

# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
print(behavioral)

# %%
conditions = behavioral["labels"]
print(conditions)

# %%
condition_mask = conditions.isin(["face", "cat"])

# %%
from nilearn.image import index_img

fmri_niimgs = index_img(fmri_filename, condition_mask)

# %%
conditions = conditions[condition_mask]
# Convert to numpy array
conditions = conditions.values
print(conditions.shape)

# %%
from nilearn.decoding import Decoder

decoder = Decoder(
    estimator="svc", mask=mask_filename, standardize="zscore_sample"
)

# %%
decoder.fit(fmri_niimgs, conditions)

# %%
prediction = decoder.predict(fmri_niimgs)
print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))

# %%
fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -30))
fmri_niimgs_test = index_img(fmri_niimgs, slice(-30, None))
conditions_train = conditions[:-30]
conditions_test = conditions[-30:]

decoder = Decoder(
    estimator="svc", mask=mask_filename, standardize="zscore_sample"
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

# %%
from sklearn.model_selection import KFold

cv = KFold(n_splits=5)

for fold, (train, test) in enumerate(cv.split(conditions), start=1):
    decoder = Decoder(
        estimator="svc", mask=mask_filename, standardize="zscore_sample"
    )
    decoder.fit(index_img(fmri_niimgs, train), conditions[train])
    prediction = decoder.predict(index_img(fmri_niimgs, test))
    print(
        "CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(
            fold,
            (prediction == conditions[test]).sum()
            / float(len(conditions[test])),
        )
    )


