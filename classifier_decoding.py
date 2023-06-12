# %%
# Model trainin and testing kobayashi et al. 2021 dataset
# Written by Merel De Merlier, Sam Vandermeulen and Max Van Migem

# %%
""" 
Load packages
"""
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import permutation_test_score

# %%
"""
Load data, initalize decoder and folds
"""

# Left and right ventral striatum
x_lvs = np.load('./train_test_data/x_lvs.npy')
x_rvs = np.load('./train_test_data/x_rvs.npy')

# Left and rigth medial orbitofrontal cortex
x_lmofc = np.load('./train_test_data/x_lmofc.npy')
x_rmofc = np.load('./train_test_data/x_rmofc.npy')

# Bahavioral labels
labels = pd.read_csv('./train_test_data/labels.csv')
labels = labels['bet_jar_type']

# Suport vector Machine decoding the default kernel is rbf
clf = SVC(kernel='linear')
cv = KFold(n_splits=5)

# %%
""" 
Model left vSTR
"""
lvs_score, lvs_perm_scores, lvs_pvalue = permutation_test_score(
    clf, x_lvs, labels, scoring="accuracy", cv=cv, n_permutations=1000
)

np.save('./model_results/lvs_perm_scores',lvs_perm_scores)
np.save('./model_results/lvs_scores',[lvs_score,lvs_pvalue])


# %%
""" 
Model right vSTR
"""
rvs_score, rvs_perm_scores, rvs_pvalue = permutation_test_score(
    clf, x_rvs, labels, scoring="accuracy", cv=cv, n_permutations=1000
)
np.save('./model_results/rvs_perm_scores.npy',rvs_perm_scores)
np.save('./model_results/rvs_scores.npy',[rvs_score,rvs_pvalue])


# %%
""" 
Models left mOFC
"""
lmofc_score, lmofc_perm_scores, lmofc_pvalue = permutation_test_score(
    clf, x_lmofc, labels, scoring="accuracy", cv=cv, n_permutations=1000
)
np.save('./model_results/lmofc_perm_scores.npy',lmofc_perm_scores)
np.save('./model_results/lmofc_scores.npy',[lmofc_score,lmofc_pvalue])


