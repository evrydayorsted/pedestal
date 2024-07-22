##### In this script, we make simple models for KBM stability for pedestal width and height across plasma squareness.

##### We read in the distance from the GCP first stability boundary and fit to models.

#### Remove alpha from the model

### NSTX

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

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']


class MidpointNormalize(mcolors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		mcolors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		normalized_min = max(0, 1 / 2 * (1 - abs((self.vmin - self.midpoint) / (self.midpoint - self.vmax))))
		normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
		normalized_mid = 0.5
		value = np.ma.masked_array(value, np.isnan(value))
		result, is_scalar = self.process_value(value)
		self.autoscale_None(result)
		result = np.ma.array(np.interp(result, [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]))
		if is_scalar:
			result = result[0]
		return result


#### These are the main arrays that we work with
squareness_flat = []
distance_to_stab_boundary_flat = []
gds2_flat = []
gds21_flat = []
gds22_flat = []
cvdrift_flat = []
cvdrift_m_gbdrift_flat = []
cvdrift0_flat = []
gradpar_flat = []
# alpha_flat = []
Bval_flat = []
beta_ped_flat = []
width_flat = []


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



######## --------- read in data from various squareness values


def reject_outliers(data, m = 2.,filtertype='mask', correlated_array = None):
		# technique for rejecting outliers based on median
		d = np.abs(data - np.median(data))
		mdev = np.median(d)
		s = d/mdev if mdev else np.zeros(len(d))
		if filtertype == 'mask':
				data2 = np.ma.array(data, mask=s>m)
				return data2
		if filtertype == 'mask_dual': # masks a second array, correlated_array, based on masking of first
				data2 = np.ma.array(data, mask=s>m)
				# ~ print('oo, s is {}'.format(s))
				# ~ print('max(s) {}'.format(np.nanmax(s)))
				# ~ print('m is {}'.format(m))
				# ~ print(*correlated_array, sep=', ')
				# ~ print('s>m is {}'.format(s>m))
				correlated_array2 = np.ma.array(correlated_array, mask=s>m)
				return data2, correlated_array2
		elif filtertype == 'reduced':
				return data[s<m]
		elif filtertype == 'reduced_dual':
				return data[s<m], correlated_array[s<m]

def read_in_to_main_arrays(equilibrium_it,file_location,square_val):
	# all_arrays = np.load('/Users/jparisi/Desktop/kbm_first_stab_nstx_zeta0p2.npz')
	all_arrays = np.load(file_location)
	beta_ped_array = all_arrays['beta_ped_array']
	width_array = all_arrays['width_array']
	cvdrift_av_array=all_arrays['cvdrift_av_array']
	cvdrift_m_gbdrift_av_array=all_arrays['cvdrift_m_gbdrift_av_array']
	cvdrift0_av_array=all_arrays['cvdrift0_av_array']
	gds2_av_array=all_arrays['gds2_av_array']
	gds21_av_array=all_arrays['gds21_av_array']
	gds22_av_array=all_arrays['gds22_av_array']
	bmag_av_array=all_arrays['bmag_av_array']
	gradpar_av_array=all_arrays['gradpar_av_array']
	# alpha_array=all_arrays['alpha_array']
	distance_to_stab_boundary_out=all_arrays['distance_to_stab_boundary']
	fixed_its=all_arrays['fixed_its']
	width=all_arrays['width']
	press=all_arrays['press']

	print(fixed_its)

	for fixed_it in fixed_its:  # April 23 new: fixed it == 0 is fixed temp, == 1 is fixed dens.
		for width_it in np.arange(len(width)):
			for press_it in np.arange(len(press)):
				if distance_to_stab_boundary_out[width_it, press_it] != 0:
					if np.isnan(distance_to_stab_boundary_out[width_it, press_it]) == False:
						distance_to_stab_boundary_flat.append(distance_to_stab_boundary_out[width_it, press_it])
						gds2_flat.append(gds2_av_array[fixed_it, width_it, press_it])
						gds21_flat.append(gds21_av_array[fixed_it, width_it, press_it])
						gds22_flat.append(gds22_av_array[fixed_it, width_it, press_it])
						cvdrift_flat.append(cvdrift_av_array[fixed_it, width_it, press_it])
						cvdrift_m_gbdrift_flat.append(cvdrift_m_gbdrift_av_array[fixed_it, width_it, press_it])
						cvdrift0_flat.append(cvdrift0_av_array[fixed_it, width_it, press_it])
						Bval_flat.append(bmag_av_array[fixed_it, width_it, press_it])
						gradpar_flat.append(gradpar_av_array[fixed_it, width_it, press_it])
						# alpha_flat.append(alpha_array[fixed_it, width_it, press_it])
						squareness_flat.append(square_val)
						beta_ped_flat.append(beta_ped_array[width_it, press_it])
						width_flat.append(width_array[width_it, press_it])
	return


def read_in_to_separate_arrays(file_location,square_val):

	squareness_flat_ = []
	distance_to_stab_boundary_flat_ = []
	gds2_flat_ = []
	gds21_flat_ = []
	gds22_flat_ = []
	cvdrift_flat_ = []
	cvdrift_m_gbdrift_flat_ = []
	cvdrift0_flat_ = []
	gradpar_flat_ = []
	# alpha_flat_ = []
	Bval_flat_ = []
	beta_ped_flat_ = []
	width_flat_ = []

	all_arrays = np.load(file_location)
	beta_ped_array = all_arrays['beta_ped_array']
	width_array = all_arrays['width_array']
	cvdrift_av_array=all_arrays['cvdrift_av_array']
	cvdrift_m_gbdrift_av_array=all_arrays['cvdrift_m_gbdrift_av_array']
	cvdrift0_av_array=all_arrays['cvdrift0_av_array']
	gds2_av_array=all_arrays['gds2_av_array']
	gds21_av_array=all_arrays['gds21_av_array']
	gds22_av_array=all_arrays['gds22_av_array']
	bmag_av_array=all_arrays['bmag_av_array']
	gradpar_av_array=all_arrays['gradpar_av_array']
	# alpha_array=all_arrays['alpha_array']
	distance_to_stab_boundary_out=all_arrays['distance_to_stab_boundary']
	fixed_its=all_arrays['fixed_its']
	width=all_arrays['width']
	press=all_arrays['press']

	print(fixed_its)

	for fixed_it in fixed_its:  # April 23 new: fixed it == 0 is fixed temp, == 1 is fixed dens.
		for width_it in np.arange(len(width)):
			for press_it in np.arange(len(press)):
				if distance_to_stab_boundary_out[width_it, press_it] != 0:
					if np.isnan(distance_to_stab_boundary_out[width_it, press_it]) == False:
						distance_to_stab_boundary_flat_.append(distance_to_stab_boundary_out[ width_it, press_it])
						gds2_flat_.append(gds2_av_array[fixed_it, width_it, press_it])
						gds21_flat_.append(gds21_av_array[fixed_it, width_it, press_it])
						gds22_flat_.append(gds22_av_array[fixed_it, width_it, press_it])
						cvdrift_flat_.append(cvdrift_av_array[fixed_it, width_it, press_it])
						cvdrift_m_gbdrift_flat_.append(cvdrift_m_gbdrift_av_array[fixed_it, width_it, press_it])
						cvdrift0_flat_.append(cvdrift0_av_array[fixed_it, width_it, press_it])
						Bval_flat_.append(bmag_av_array[fixed_it, width_it, press_it])
						gradpar_flat_.append(gradpar_av_array[fixed_it, width_it, press_it])
						# alpha_flat_.append(alpha_array[fixed_it, width_it, press_it])
						squareness_flat_.append(square_val)
					beta_ped_flat_.append(beta_ped_array[width_it, press_it])
					width_flat_.append(width_array[width_it, press_it])
	# return [squareness_flat_, distance_to_stab_boundary_flat_, gds2_flat_, gds21_flat_, gds22_flat_, cvdrift_flat_, cvdrift_m_gbdrift_flat_, cvdrift0_flat_, gradpar_flat_, alpha_flat_, Bval_flat_, beta_ped_flat_, width_flat_]
	return [squareness_flat_, distance_to_stab_boundary_flat_, gds2_flat_, gds21_flat_, gds22_flat_, cvdrift_flat_, cvdrift_m_gbdrift_flat_, cvdrift0_flat_, gradpar_flat_, Bval_flat_, beta_ped_flat_, width_flat_]


def return_variable(file_location, var_to_return='beta_ped_array'):
	all_arrays = np.load(file_location)
	return all_arrays[var_to_return]

remove_correlations = True # if True, removes highly correlated variables from analysis

n_squareness = 1

# geo_quants_full = [r'$\langle | \nabla_N \alpha |^2 \rangle$',r'$\langle \nabla_N \alpha \cdot \nabla_N q \rangle$',r'$\langle | \nabla_N q |^2 \rangle$',r'$\langle \omega_{\kappa}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{\alpha}-\omega_{\nabla B}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{q} \rangle$',r'$\langle \mathrm{B} \rangle$', r'$\langle \hat{b} \cdot \nabla_N \theta \rangle$', r'$\alpha_{\mathrm{MHD}}$']
# geo_quants = [r'$\langle | \nabla_N \alpha |^2 \rangle$',r'$\langle \nabla_N \alpha \cdot \nabla_N q \rangle$',r'$\langle | \nabla_N q |^2 \rangle$',r'$\langle \omega_{\kappa}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{\alpha}-\omega_{\nabla B}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{q} \rangle$',r'$\langle \mathrm{B} \rangle$', r'$\langle \hat{b} \cdot \nabla_N \theta \rangle$', r'$\alpha_{\mathrm{MHD}}$']

geo_quants_full = [r'$\langle | \nabla_N \alpha |^2 \rangle$',r'$\langle \nabla_N \alpha \cdot \nabla_N q \rangle$',r'$\langle | \nabla_N q |^2 \rangle$',r'$\langle \omega_{\kappa}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{\alpha}-\omega_{\nabla B}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{q} \rangle$',r'$\langle \mathrm{B} \rangle$', r'$\langle \hat{b} \cdot \nabla_N \theta \rangle$']
geo_quants = [r'$\langle | \nabla_N \alpha |^2 \rangle$',r'$\langle \nabla_N \alpha \cdot \nabla_N q \rangle$',r'$\langle | \nabla_N q |^2 \rangle$',r'$\langle \omega_{\kappa}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{\alpha}-\omega_{\nabla B}^{\alpha} \rangle$',r'$\langle \omega_{\kappa}^{q} \rangle$',r'$\langle \mathrm{B} \rangle$', r'$\langle \hat{b} \cdot \nabla_N \theta \rangle$']

ngeo_coefficients = len(geo_quants) # number of coefficients corresponding to below
label_size = 40

######### MAST U 48339
# # squareness_0p5 loading
# square_val = 0.5
# read_in_to_main_arrays(equilibrium_it=0,file_location = '/Users/jparisi/Desktop/MASTU_48339_zeta0p5_machine_learn4.npz',square_val=square_val)

######### NSTX 132543
square_val = 0.27
read_in_to_main_arrays(equilibrium_it=0,file_location = 'nstx_132543_zeta0p27_save_machine_learn4.npz',square_val=square_val)

######## --------- correlation coefficients
if len(distance_to_stab_boundary_flat) > 1:
	# quantities = [gds2_flat, gds21_flat, gds22_flat, cvdrift_flat, cvdrift_m_gbdrift_flat, cvdrift0_flat, Bval_flat, gradpar_flat, alpha_flat]  # Add more as needed
	quantities = [gds2_flat, gds21_flat, gds22_flat, cvdrift_flat, cvdrift_m_gbdrift_flat, cvdrift0_flat, Bval_flat, gradpar_flat]  # Add more as needed
	corr_array = np.zeros(ngeo_coefficients) #
	counter = 0
	for i, qty in enumerate(quantities, 1):
			s_corr, _ = spearmanr(distance_to_stab_boundary_flat, qty)
			print(f"Correlations for quantity {i}: Spearman={s_corr}")
			corr_array[counter] = s_corr
			counter = counter + 1

	fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = .5, wspace=.7)
	axs.bar(geo_quants, corr_array[:])
	axs.set_ylim(-1,1)
	axs.set_xlabel('geo quantity', fontsize = label_size)
	axs.set_xticklabels(geo_quants, rotation=65)
	axs.set_ylabel('Spearman correlation', fontsize = label_size)
	plt.suptitle(r'Correlation w/ $\delta \beta_{{\theta,\mathrm{{ped}}}}$ for KBM, sample size = {}'.format(len(distance_to_stab_boundary_flat)), fontsize = 0.6*label_size)
	plt.savefig("pre_filtered_pre_correlation_removal_Spearman_deltabetaped_nstx.pdf",bbox_inches='tight', pad_inches=0.1)

	# plt.show()


mval = 5 # Filter out everything more than 5 medians from median...
quantities_filtered = []
counter = 0
for i, qty in enumerate(quantities, 1):
	# Filter qty
	distance_to_stab_boundary_flat_filter, qty_filter  = reject_outliers(np.array(distance_to_stab_boundary_flat), m = mval,filtertype='reduced_dual', correlated_array = np.array(qty))
	quantities_filtered.append(qty_filter)
X = np.vstack(quantities_filtered).T
y = distance_to_stab_boundary_flat_filter
degree = 2 # quadratic model

if remove_correlations == True:

	# Remove highly correlated features
	X_cleaned, removed_features = remove_highly_correlated_features(X, threshold=0.8)
	print("Removed features indices:", removed_features)
	X = X_cleaned

	### need to update geo quants
	geo_quants_update = []
	for geo_it in np.arange(len(geo_quants)):
		if geo_it not in removed_features:
			geo_quants_update.append(geo_quants[geo_it])
	geo_quants = geo_quants_update

# Create a pipeline with standardization and a non-linear model
model = make_pipeline(StandardScaler(),       # Step 1: Standardize the features
	PolynomialFeatures(degree),  # Step 2: Generate polynomial features (optional, based on model needs)
	RandomForestRegressor())  # Step 3: Fit a non-linear regression model

# Fit the model
model.fit(X, y)
# Calculate R^2
r2 = model.score(X, y)
print("R^2 random forest:", r2) #

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

# Mapping feature names to geo_quants
variable_mapping = {f'x{i}': name for i, name in enumerate(geo_quants)}
# Function to replace variables in composite names
def replace_feature_names(feature_name, mapping):
        pattern = re.compile(r'x\d+')
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
# plt.show()
#plt.savefig("feature_importance_deltabetaped_nstx.pdf",bbox_inches='tight', pad_inches=0.1)


#### PERMUTATION IMPORTANCE

degree = 2

# Create a pipeline with standardization, polynomial features, and Random Forest
model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), RandomForestRegressor())

# Fit the model
model.fit(X, y)

# Calculate R^2
r2 = model.score(X, y)
print("R^2:", r2)

# Compute permutation importance
result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='r2')

# Get feature names
poly_feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out()

# Exclude the constant term '1'
important_features = [name for name in poly_feature_names if name != '1']

# Get the permutation importance results
importances_mean = result.importances_mean
importances_std = result.importances_std

# Create a sorted list of features based on importance
sorted_idx = importances_mean.argsort()

# Plot the permutation importance
plt.figure(figsize=(10, 6))
sorted_important_features = np.array(geo_quants)[sorted_idx]
sorted_importances_mean = importances_mean[sorted_idx]
bars=plt.barh(range(len(sorted_importances_mean)), sorted_importances_mean, xerr=importances_std[sorted_idx])
plt.yticks(range(len(importances_mean[sorted_idx])), sorted_important_features)

plt.xlabel(r"Permutation Importance")
plt.title(r"Permutation Importances in Random Forest Model for $\delta \beta_{\theta,\mathrm{ped}}$")
# plt.show()
#plt.savefig("permutation_importance_deltabetaped_nstx.pdf",bbox_inches='tight', pad_inches=0.1)



######## --------- fit to machine learning model





######## --------- assess feature and permutation importance




######## --------- assess correlations between x variables we are fitting to...


data = {
    'gds2': gds2_flat,
    'gds21': gds21_flat,
    'gds22': gds22_flat,
    'cvdrift': cvdrift_flat,
    'cvdrift_m_gbdrift_flat': cvdrift_m_gbdrift_flat,
    'cvdrift0_flat': cvdrift0_flat,
    'Bval_flat': Bval_flat,
    'gradpar_flat': gradpar_flat,
    # 'alpha_flat': alpha_flat,
}

df = pd.DataFrame(data)


# Compute the Spearman correlation matrix
spearman_corr = df.corr(method='spearman')

# Zero out the upper triangle of the correlation matrix, excluding the diagonal
mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
spearman_corr.values[mask] = np.nan  # Using NaN to avoid displaying zero and to avoid confusion with actual zero correlations

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
ax=sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,xticklabels=geo_quants_full, yticklabels=geo_quants_full)
ax.set_yticklabels(ax.get_yticklabels(),rotation=00)
plt.title('Spearman Correlation Matrix')
plt.show()





######## --------- Read in delta beta as a function of shaping!!

def read_in_to_separate_arrays_shaping(file_location):

	ar_flat_ = []
	elongation_flat_ = []
	squareness_flat_ = []
	triangularity_flat_ = []
	gds2_flat_ = []
	gds21_flat_ = []
	gds22_flat_ = []
	cvdrift_flat_ = []
	cvdrift_m_gbdrift_flat_ = []
	cvdrift0_flat_ = []
	gradpar_flat_ = []
	# alpha_flat_ = []
	Bval_flat_ = []

	all_arrays = np.load(file_location)
	cvdrift_av_array=all_arrays['cvdrift_av_array']
	cvdrift_m_gbdrift_av_array=all_arrays['cvdrift_m_gbdrift_av_array']
	cvdrift0_av_array=all_arrays['cvdrift0_av_array']
	gds2_av_array=all_arrays['gds2_av_array']
	gds21_av_array=all_arrays['gds21_av_array']
	gds22_av_array=all_arrays['gds22_av_array']
	bmag_av_array=all_arrays['bmag_av_array']
	gradpar_av_array=all_arrays['gradpar_av_array']
	# alpha_array=all_arrays['alpha_array']
	elon_array=all_arrays['elon_array']
	tri_array=all_arrays['tri_array']
	square_array=all_arrays['square_array']
	AR_array=all_arrays['AR_array']

	num_kappa_scans = len(elon_array)
	num_tri_scans = len(tri_array)
	num_square_scans = len(square_array)
	num_AR_scans = len(AR_array)

	for kappa_it in np.arange(num_kappa_scans):
		kappa_val_here = elon_array[kappa_it]
		for tri_it in np.arange(num_tri_scans):
			tri_val_here = tri_array[tri_it]
			for square_it in np.arange(num_square_scans):
				square_val_here = square_array[square_it]
				for AR_it in np.arange(num_AR_scans):
					AR_val_here = AR_array[AR_it]
					if cvdrift_av_array[kappa_it, tri_it, square_it, AR_it] != 0:
						if np.isnan(cvdrift_av_array[kappa_it, tri_it, square_it, AR_it]) == False:
							gds2_flat_.append(gds2_av_array[kappa_it, tri_it, square_it, AR_it])
							gds21_flat_.append(gds21_av_array[kappa_it, tri_it, square_it, AR_it])
							gds22_flat_.append(gds22_av_array[kappa_it, tri_it, square_it, AR_it])
							cvdrift_flat_.append(cvdrift_av_array[kappa_it, tri_it, square_it, AR_it])
							cvdrift_m_gbdrift_flat_.append(cvdrift_m_gbdrift_av_array[kappa_it, tri_it, square_it, AR_it])
							cvdrift0_flat_.append(cvdrift0_av_array[kappa_it, tri_it, square_it, AR_it])
							Bval_flat_.append(bmag_av_array[kappa_it, tri_it, square_it, AR_it])
							gradpar_flat_.append(gradpar_av_array[kappa_it, tri_it, square_it, AR_it])
							ar_flat_.append(AR_array[AR_it])
							elongation_flat_.append(elon_array[kappa_it])
							squareness_flat_.append(square_array[square_it])
							triangularity_flat_.append(tri_array[tri_it])
							# alpha_flat_.append(alpha_array[kappa_it, tri_it, square_it, AR_it])

	# return [gds2_flat_, gds21_flat_, gds22_flat_, cvdrift_flat_, cvdrift_m_gbdrift_flat_, cvdrift0_flat_, gradpar_flat_, alpha_flat_, Bval_flat_, ar_flat_, elongation_flat_, triangularity_flat_, squareness_flat_]
	return [gds2_flat_, gds21_flat_, gds22_flat_, cvdrift_flat_, cvdrift_m_gbdrift_flat_, cvdrift0_flat_, gradpar_flat_, Bval_flat_, ar_flat_, elongation_flat_, triangularity_flat_, squareness_flat_]

## Need to import the geometric coefficients for each plasma shape scan.
nstx_132543_shaping_scan = read_in_to_separate_arrays_shaping('nstx132543_shape_scan_save_geomachine_learn1.npz')

###### Now evaluate the distance to the first stability boundary for these different shapes! Cool.

# quantities_new = np.array([nstx_132543_shaping_scan[0], nstx_132543_shaping_scan[1], nstx_132543_shaping_scan[2], nstx_132543_shaping_scan[3], nstx_132543_shaping_scan[4], nstx_132543_shaping_scan[5], nstx_132543_shaping_scan[6], nstx_132543_shaping_scan[7], nstx_132543_shaping_scan[8]])  # Add more as needed
quantities_new = np.array([nstx_132543_shaping_scan[0], nstx_132543_shaping_scan[1], nstx_132543_shaping_scan[2], nstx_132543_shaping_scan[3], nstx_132543_shaping_scan[4], nstx_132543_shaping_scan[5], nstx_132543_shaping_scan[6], nstx_132543_shaping_scan[7]])  # Add more as needed
X = np.vstack(quantities_new).T

## let's sort the test data according to the required format!

new_test_data_cleaned = np.delete(quantities_new, removed_features, axis=0)

# Make predictions
predictions = model.predict(new_test_data_cleaned.T)

# Print predictions
print("Predictions for new test data:")
print(predictions)

x = nstx_132543_shaping_scan[-2]
y = nstx_132543_shaping_scan[-1]

xi = np.linspace(np.min(x), np.max(x), 100)
yi = np.linspace(np.min(y), np.max(y), 100)

xi, yi = np.meshgrid(xi, yi)

predictions_interpolated = griddata((x, y), predictions, (xi, yi), method='cubic')

# Apply Gaussian smoothing
zi_smooth = gaussian_filter(predictions_interpolated, sigma=5.5)
norm = MidpointNormalize(vmin=zi_smooth.min(),vmax=zi_smooth.max(),midpoint=0)

###### Regular colorscheme
fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.7)
# scatter =ax.scatter(nstx_132543_shaping_scan[-2],nstx_132543_shaping_scan[-1],c=predictions)
contour =ax.contourf(xi,yi,zi_smooth,cmap='seismic', norm = norm)
cbar = plt.colorbar(contour)  # Show color scale
cbar.ax.tick_params(labelsize=24)
cbar.set_label(r'$\delta \beta_{\theta,\mathrm{ped}}$', fontsize=label_size)
ax.set_xlabel(r'$\delta_0$', fontsize = label_size)
ax.set_ylabel(r'$\zeta_0$', fontsize = label_size)
plt.tick_params(axis='both', which='major', labelsize=24) 
plt.show()
####### Dark color scheme
plt.style.use('dark_background')  # Set the style to dark background
fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.7)
# scatter =ax.scatter(nstx_132543_shaping_scan[-2],nstx_132543_shaping_scan[-1],c=predictions)
ax.contour(xi,yi,zi_smooth,levels= [0],colors=['k'], linestyles = ['--'], linewidths = [3])
contour =ax.contourf(xi,yi,zi_smooth,cmap='seismic', norm = norm)
cbar = plt.colorbar(contour)  # Show color scale
cbar.ax.tick_params(labelsize=24)
cbar.set_label(r'$\delta \beta_{\theta,\mathrm{ped}}$', fontsize=label_size)
# ax.set_xlabel(r'$\delta_0$', fontsize = label_size, color='white')
# ax.set_ylabel(r'$\zeta_0$', fontsize = label_size, color='white')
ax.set_ylabel('squareness', fontsize = label_size, color='white')
ax.set_xlabel('triangularity', fontsize = label_size, color='white')
plt.tick_params(axis='both', which='major', labelsize=24, colors='white') 
# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
# plt.show()
plt.savefig('fig1.pdf')
plt.close()


####### ---------- probably need to test feature and permutation on TEST data. So, train on training data, then test on test data! Cool.








