"""
Adapted from Jason Parisi machine_learning_local_prv.py
"""



from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.pyplot import cm
# import netCDF4 as nc4
from scipy.interpolate import interp1d
import seaborn as sns
from copy import deepcopy
from scipy.stats import spearmanr
from scipy.integrate import simps
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path
import time as timetime
import numpy as np
import re
import matplotlib.colors as mcolors
from pedestal import *



def remove_highly_correlated_features(X, threshold=0.8):
    """
    Remove features that are highly correlated with other features.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - threshold: correlation threshold for identifying highly correlated features
    
    Returns:
    - X_cleaned: numpy array with highly correlated features removed
    - removed_features: list of indices of features that were removed
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    # Select upper triangle of correlation matrix
    upper = np.triu(corr_matrix, k=1)
    
    # Find features with correlation greater than the threshold
    to_drop = [i for i in range(upper.shape[0]) if any(upper[i, :] > threshold)]
    
    # Drop features
    X_cleaned = np.delete(X, to_drop, axis=1)
    
    return X_cleaned, to_drop


# load pedestal data from pkl file
a = Shot("allShots", "pkl")


# Create a pipeline with standardization and a non-linear model
model = make_pipeline(StandardScaler(),  # Step 2: Generate polynomial features (optional, based on model needs)
	PolynomialFeatures(1), 
    RandomForestRegressor())  # Step 3: Fit a non-linear regression model

#extract pedestal parameters to fit
X = np.vstack((a.H_ped_psin_ne,a.H_ped_psin_pe,a.H_ped_psin_te,a.H_ped_radius_ne,a.H_ped_radius_pe,a.H_ped_radius_te,a.Ip, a.W_ped_psin_ne,a.W_ped_psin_pe,a.W_ped_psin_te,a.W_ped_radius_ne,a.W_ped_radius_pe,a.W_ped_radius_te, a.W_ped,a.aratio, a.beamPower, a.betaN, a.delta, a.elmPercent, a.elong, a.greenwaldFraction, a.shotIndexed, a.ssNBI, a.swNBI, a.times)).T


# Remove highly correlated features
X=X.astype("float64")
X=np.nan_to_num(X, nan=0)
X[np.where(np.isnan(X))]=0
X_cleaned, removed_features = remove_highly_correlated_features(X, threshold=0.8)
print("Removed features indices:", removed_features)

X = X_cleaned

#pedestal height is parameter to predict
y = a.Beta_ped

# attempts to fix NaN error
print(X[np.where(np.isinf(X))])
for i in range(len(X)):
    X[i][np.where(np.isnan(X[i]))]=0
y[np.where(np.isnan(y))]=0

print("yep",X[np.where(np.isnan(X))])
X.T[0] =X.T[0]/1e18
print(X)
print(np.where(np.isnan(y)==True))
# for i in range(len(X.T)):
#     z = np.isfinite(X.T[i])
#     print(np.where(z==False))
print(X.shape, y.shape)
print(y)
print(X.dtype)
print(y.dtype)


#fit model
model.fit(X, y)
# Calculate R^2
r2 = model.score(X, y)
print("R^2 random forest:", r2) #
