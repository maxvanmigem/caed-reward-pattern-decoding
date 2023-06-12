# case-studies-mvpa
Repository of a group project for the course: 'Case studies in the Analysis of Experimental Data'. This project entails the analysis of fMRI data using pattern decoding analysis.

## About
The present study aims to investigate how the vSTR and the mOFC each contribute to the encoding of information related to the anticipation of reward. To do this we use pattern decoding to see how activity in these areas can predict exposure to different levels of expected reward magnitude. More specifically, we train multiple classifiers using data from either the vSTR or the mOFC and compare their respective accuracies for predicting low or high expected reward.  

The dataset can be found here: https://openneuro.org/datasets/ds003758/versions/1.0.2
The origanal study: https://doi.org/10.1523/JNEUROSCI.0423-21.2021
The regions of intrest (ROI) were gathered from the Julich-Brain Atlas v3.01 (https://search.kg.ebrains.eu/instances/99437b56-3add-4d19-9bfd-5b413dbf5173)

## Team
The team consists of: Merel De Merlier, Sam Vandermeulen and Maximilien Van Migem

## Setup & Requirements
Python (v3.9.7)
Nilearn package (v0.10.1) + dependencies

## Order of opperations
1. roi_gen.py file - used to adapt the atlas images to fit the needs of the current study and plot
2. feature_selection.py file - averages func images across trials and aplies spatial masks so that it is ready to be given to the classifiers
3. classifier_decoding.py file - train and test classifier + permutation test
4. result_plotting.py file - plots results

