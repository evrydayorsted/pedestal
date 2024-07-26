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


######---- begin jfp edits

# Create a pipeline with standardization and a non-linear model
model = make_pipeline(StandardScaler(),  # Step 2: Generate polynomial features (optional, based on model needs)
	PolynomialFeatures(2), 
    RandomForestRegressor())  # Step 3: Fit a non-linear regression model

#extract pedestal parameters to fit
quantities_filtered = []
n_quantities = 25

quantities_filtered.append(a.H_ped_psin_ne)
quantities_filtered.append(a.H_ped_psin_pe)
quantities_filtered.append(a.H_ped_psin_te)
quantities_filtered.append(a.H_ped_radius_ne)
quantities_filtered.append(a.H_ped_radius_pe)
quantities_filtered.append(a.H_ped_radius_te)
quantities_filtered.append(a.Ip)
quantities_filtered.append(a.W_ped_psin_ne)
quantities_filtered.append(a.W_ped_psin_pe)
quantities_filtered.append(a.W_ped_psin_te)
quantities_filtered.append(a.W_ped_radius_ne)
quantities_filtered.append(a.W_ped_radius_pe)
quantities_filtered.append(a.W_ped_radius_te)
quantities_filtered.append(a.W_ped)
quantities_filtered.append(a.aratio)
quantities_filtered.append(np.array(a.beamPower,dtype=float)) # cast to float
quantities_filtered.append(a.betaN)
quantities_filtered.append(a.delta)
quantities_filtered.append(a.elmPercent) # NaNs and Infs here
quantities_filtered.append(a.elong)
quantities_filtered.append(a.greenwaldFraction) # NaNs and Infs here
quantities_filtered.append(a.shotIndexed)
quantities_filtered.append(a.ssNBI)
quantities_filtered.append(a.swNBI)
quantities_filtered.append(a.times)

X_temp = np.vstack(quantities_filtered).T
y = a.Beta_ped
# e.g. there is a NaN in 42557 due to elmPercent -- see X_temp[42557]

# check for any NaN vlalues
nan_mask = ~np.isnan(X_temp).any(axis=1)
# filter out rows w/ NaN values
X = X_temp[nan_mask]
y = y[nan_mask]

##### let's subsample for a quick model training
X_sub = X[::100]
y_sub = y[::100]
model.fit(X_sub, y_sub) ## training in ~5 seconds
# Calculate R^2
r2 = model.score(X_sub, y_sub)
print("R^2 random forest:", r2) #






# ######---- end jfp edits

# # Remove highly correlated features
# X=X.astype("float64")
# X=np.nan_to_num(X, nan=0)
# X[np.where(np.isnan(X))]=0
# X_cleaned, removed_features = remove_highly_correlated_features(X, threshold=0.8)
# print("Removed features indices:", removed_features)

# X = X_cleaned

# #pedestal height is parameter to predict
# y = a.Beta_ped

# # attempts to fix NaN error
# print(X[np.where(np.isinf(X))])
# for i in range(len(X)):
#     X[i][np.where(np.isnan(X[i]))]=0
# y[np.where(np.isnan(y))]=0

# print("yep",X[np.where(np.isnan(X))])
# X.T[0] =X.T[0]/1e18
# print(X)
# print(np.where(np.isnan(y)==True))
# # for i in range(len(X.T)):
# #     z = np.isfinite(X.T[i])
# #     print(np.where(z==False))
# print(X.shape, y.shape)
# print(y)
# print(X.dtype)
# print(y.dtype)


# #fit model
# model.fit(X, y)


#### FEATURE IMPORTANCE

# Get the RandomForestRegressor from the pipeline
rf_model = model.named_steps['randomforestregressor']

# Feature importances
feature_importances = rf_model.feature_importances_

# Feature names
poly_feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out()
feature_importance_dict = dict(zip(poly_feature_names, feature_importances))

# Sort features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

# Print sorted features with importances
print("Feature importances (sorted):")
for name, importance in sorted_features:
        print(f"Feature: {name}, Importance: {importance}")

# Sort features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
feature_names_sorted = [item[0] for item in sorted_features]
importances_sorted = [item[1] for item in sorted_features]


pedQuants = [r'$\langle | \nabla_N \alpha |^2 \rangle$',r'$\langle \nabla_N \alpha \cdot \nabla_N q \rangle$',r'$\langle | \nabla_N q |^2 \rangle$',r'$\langle \omega_{\kappa}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{\alpha}-\omega_{\nabla B}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{q} \rangle$',r'$\langle \mathrm{B} \rangle$', r'$\langle \hat{b} \cdot \nabla_N \theta \rangle$']

# Mapping feature names to geo_quants
variable_mapping = {f'x{i}': name for i, name in enumerate(pedQuants)}
# Function to replace variables in composite names
def replace_feature_names(feature_name, mapping):
        pattern = re.compile(r'x\d+')
        print("hello")
        return pattern.sub(lambda x: mapping[x.group()], feature_name)
# Replacing feature names with the actual variable names
feature_names_latex = [replace_feature_names(name, variable_mapping) for name in feature_names_sorted]

# Plot feature importances --- pick only top 7 variables!
top_filter = 7
plt.figure(figsize=(10, 6))
plt.barh(feature_names_latex[:top_filter], importances_sorted[:top_filter], color='b', align='center')
# ~ plt.barh(feature_names_sorted[:top_filter], importances_sorted[:top_filter], color='b', align='center')
# ~ plt.barh(feature_names_sorted[-top_filter:], importances_sorted[-top_filter:], color='b', align='center')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest Model For $\delta \beta_{\theta,\mathrm{ped}}$')
plt.gca().invert_yaxis()
plt.show()
#plt.savefig("feature_importance_deltabetaped_nstx.pdf",bbox_inches='tight', pad_inches=0.1)


