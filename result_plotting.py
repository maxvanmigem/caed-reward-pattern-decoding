# %%
# Reward decoding analysis kobayashi et al. 2021 dataset
# Written by Merel De Merlier, Sam Vandermeulen and Max Van Migem

# %%
""" 
Load packages
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

# %%
"""
Load results
"""
# right vSTR
rvs_perm_scores = np.load('./model_results/rvs_perm_scores.npy')
rvs_scores = np.load('./model_results/rvs_scores.npy')

# Left vSTR
lvs_perm_scores = np.load('./model_results/lvs_perm_scores.npy')
lvs_scores = np.load('./model_results/lvs_scores.npy')
# Bilateral vSTR
bivs_perm_scores = np.load('./model_results/bivs_perm_scores.npy')
bivs_scores = np.load('./model_results/bivs_scores.npy')
# Left mOFC
lmofc_perm_scores = np.load('./model_results/lmofc_perm_scores.npy')
lmofc_scores = np.load('./model_results/lmofc_scores.npy')

# %%
""" 
a function that doesn't work
"""
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

   
    text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

 
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

# %%
my_palette = sns.color_palette("husl", 9).as_hex()
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [lmofc_scores[0], lvs_scores[0], rvs_scores[0],bivs_scores[0]]
 
# Choose the height of the cyan bars
bars2 = [lmofc_perm_scores.mean(), lvs_perm_scores.mean(), rvs_perm_scores.mean(), bivs_perm_scores.mean()]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = my_palette[1], edgecolor = 'black', capsize=5, label='Classifier')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = my_palette[7], edgecolor = 'black', capsize=7, label='Permutation test')
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], ['Left mOFC', 'Left vSTR', 'Right vSTR', 'Bilateral vSTR'])
plt.ylabel('Accuracy')
plt.legend()

plt.ylim(0.7, 0.8)
barplot_annotate_brackets(0, 1, .1, [bars1[0],bars2[0]], [0.75,0.75])
# Show graphic
plt.show()

# %%



