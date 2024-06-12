IMPORT AS 
midradius_idx = nradial_locs // 2

ticklabelsize = 12
label_size = 20

plt.rcParams['xtick.labelsize'] = ticklabelsize
plt.rcParams['ytick.labelsize'] = ticklabelsize

fixed_it = 1

theta_min_av_val = -3 * np.pi
theta_max_av_val = 3 * np.pi

#### Jun 6 2024: adding more stuff to the model! alpha and shat, for example!

geo_quants = ['$\\langle \\mathrm{{gds2}} \\rangle$','$\\langle \\mathrm{{gds21}} \\rangle$','$\\langle \\mathrm{{gds22}} \\rangle$','$\\langle \\mathrm{{cvd}} \\rangle$','$\\langle \\mathrm{{cvd-gbd}} \\rangle$','$\\langle \\mathrm{{cvd0}} \\rangle$','$\\langle \\mathrm{{B}} \\rangle$']
# ~ geo_quants = ['$\\langle \\mathrm{{gds2}} \\rangle$','$\\langle \\mathrm{{gds21}} \\rangle$','$\\langle \\mathrm{{gds22}} \\rangle$','$\\langle \\mathrm{{cvd}} \\rangle$','$\\langle \\mathrm{{B}} \\rangle$']
from scipy.stats import spearmanr
ngeo_coefficients = len(geo_quants) # number of coefficients corresponding to below
# eigenmode-averaged coefficients
cvdrift_av_array, cvdrift_m_gbdrift_av_array, cvdrift0_av_array, gds2_av_array, gds21_av_array, gds22_av_array, bmag_av_array = readin_basic_geo_data(theta_min_av_val, theta_max_av_val, calc_mode='average_eigenmode', permitted_mode_array_in=permitted_mode_array)
# gamma / kperp2 
gamma_over_kperp2_array_out = readin_basic_geo_data(theta_min_av_val, theta_max_av_val, calc_mode='average_eigenmode_gamma_kperp2', permitted_mode_array_in=permitted_mode_array)

gamma_over_kperp2_nonzero_flat = []
gds2_flat = []
gds21_flat = []
gds22_flat = []
cvdrift_flat = []
cvdrift_m_gbdrift_flat = []
cvdrift0_flat = []
Bval_flat = []

for fixed_it in root['INPUTS']['input']['general']['fixed_its']:  # April 23 new: fixed it == 0 is fixed temp, == 1 is fixed dens.
	for width_it in np.arange(len(width)):
		for press_it in np.arange(len(press)):
			if gamma_over_kperp2_array_out[fixed_it, width_it, press_it] != 0:
				if np.isnan(gamma_over_kperp2_array_out[fixed_it, width_it, press_it]) == False:
					gamma_over_kperp2_nonzero_flat.append(gamma_over_kperp2_array_out[fixed_it, width_it, press_it])
					gds2_flat.append(gds2_av_array[fixed_it, width_it, press_it])
					gds21_flat.append(gds21_av_array[fixed_it, width_it, press_it])
					gds22_flat.append(gds22_av_array[fixed_it, width_it, press_it])
					cvdrift_flat.append(cvdrift_av_array[fixed_it, width_it, press_it])
					cvdrift_m_gbdrift_flat.append(cvdrift_m_gbdrift_av_array[fixed_it, width_it, press_it])
					cvdrift0_flat.append(cvdrift0_av_array[fixed_it, width_it, press_it])
					Bval_flat.append(bmag_av_array[fixed_it, width_it, press_it])


if len(gamma_over_kperp2_nonzero_flat) > 1:
	quantities = [gds2_flat, gds21_flat, gds22_flat, cvdrift_flat, cvdrift_m_gbdrift_flat, cvdrift0_flat, Bval_flat]  # Add more as needed
	corr_array = np.zeros(ngeo_coefficients) #
	counter = 0
	for i, qty in enumerate(quantities, 1):
		s_corr, _ = spearmanr(gamma_over_kperp2_nonzero_flat, qty)
		print(f"Correlations for quantity {i}: Spearman={s_corr}")
		corr_array[counter] = s_corr
		counter = counter + 1

	### Also just plot spearman by itself
	fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = .5, wspace=.7)
	axs.bar(geo_quants, corr_array[:])
	axs.set_ylim(-1,1)
	axs.set_xlabel('geo quantity', fontsize = label_size)
	axs.set_xticklabels(geo_quants, rotation=65)
	axs.set_ylabel('Spearman correlation', fontsize = label_size)		
	plt.suptitle('Correlation w/ $\\gamma / k_{{\\perp}}^2$ for \n' + 'KBM' + ', sample size = {}'.format(len(gamma_over_kperp2_nonzero_flat)) + ', where $\\langle \\ldots \\rangle = \\int |\\phi|^2 \\ldots d\\theta / \\int |\\phi|^2 d\\theta$ no filter', fontsize = 0.6*label_size)	
	# ~ plt.suptitle('Correlation with $\\gamma / k_{{\\perp}}^2$ for ' + instability_array[modetype_it] + ' modes, sample size = {}'.format(len(gamma_over_kperp2_nonzero_flat)) + ', where $\\langle \\ldots \\rangle = \\int |\\phi|^2 \\ldots d\\theta / \\int |\\phi|^2 d\\theta$')	
	# ~ plt.savefig(plot_out_dir+meta_scan_name+"_spearman_coefficients_mode_"+str(instability_array[modetype_it])+".pdf",bbox_inches='tight', pad_inches=0.1)
	# ~ plt.close()
	plt.show()


# June 6 2024: question. Are we rejecting outliers? might be a good idea. And then, are we rejecting outliners in the same way for each geo quantity or differently?

# Approach 1: filter based on each individual geo quantity
m_filter_array = [2,4,10] # different filter values based on distance from median. 

for mval in m_filter_array:

	if len(gamma_over_kperp2_nonzero_flat) > 1:
		quantities = [gds2_flat, gds21_flat, gds22_flat, cvdrift_flat, cvdrift_m_gbdrift_flat, cvdrift0_flat, Bval_flat]  # Add more as needed
		corr_array_filter = np.zeros(ngeo_coefficients) #
		counter = 0
		for i, qty in enumerate(quantities, 1):
			# Filter qty
			qty_filter, gamma_over_kperp2_nonzero_flat_filter = reject_outliers(np.array(qty), m = mval,filtertype='reduced_dual', correlated_array = np.array(gamma_over_kperp2_nonzero_flat))
			s_corr, _ = spearmanr(gamma_over_kperp2_nonzero_flat_filter, qty_filter)
			print('len(qty) is {} and len(qty_filter) is {}'.format(len(qty),len(qty_filter)))
			print(f"Correlations for quantity {i}: Spearman={s_corr}")
			corr_array_filter[counter] = s_corr
			counter = counter + 1

		### Also just plot spearman by itself
		fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.7)
		axs.bar(geo_quants, corr_array_filter[:])
		axs.set_ylim(-1,1)
		axs.set_xlabel('geo quantity', fontsize = label_size)
		axs.set_xticklabels(geo_quants, rotation=65)
		axs.set_ylabel('Spearman correlation', fontsize = label_size)		
		plt.suptitle('Correlation w/ $\\gamma / k_{{\\perp}}^2$ for KBM \n' + 'where $\\langle \\ldots \\rangle = \\int |\\phi|^2 \\ldots d\\theta / \\int |\\phi|^2 d\\theta$ (w/ geo filter {})'.format(mval), fontsize = 0.6*label_size)	
		# ~ plt.suptitle('Correlation with $\\gamma / k_{{\\perp}}^2$ for ' + instability_array[modetype_it] + ' modes, sample size = {}'.format(len(gamma_over_kperp2_nonzero_flat)) + ', where $\\langle \\ldots \\rangle = \\int |\\phi|^2 \\ldots d\\theta / \\int |\\phi|^2 d\\theta$')	
		# ~ plt.savefig(plot_out_dir+meta_scan_name+"_spearman_coefficients_mode_"+str(instability_array[modetype_it])+".pdf",bbox_inches='tight', pad_inches=0.1)
		# ~ plt.close()
		plt.show()

## Second approach: we filter based on a given geometric array (or perhaps gamma/kperp2?) and then apply same filter accordingly to all other arrays. Cool.
m_filter_array = [2,4,10] # different filter values based on distance from median. 

for mval in m_filter_array:

	if len(gamma_over_kperp2_nonzero_flat) > 1:
		quantities = [gds2_flat, gds21_flat, gds22_flat, cvdrift_flat, cvdrift_m_gbdrift_flat, cvdrift0_flat, Bval_flat]  # Add more as needed
		corr_array_filter = np.zeros(ngeo_coefficients) #
		counter = 0
		for i, qty in enumerate(quantities, 1):
			# Filter qty
			# ~ qty_filter, gamma_over_kperp2_nonzero_flat_filter = reject_outliers(np.array(qty), m = mval,filtertype='reduced_dual', correlated_array = np.array(gamma_over_kperp2_nonzero_flat))
			gamma_over_kperp2_nonzero_flat_filter, qty_filter  = reject_outliers(np.array(gamma_over_kperp2_nonzero_flat), m = mval,filtertype='reduced_dual', correlated_array = np.array(qty))
			s_corr, _ = spearmanr(gamma_over_kperp2_nonzero_flat_filter, qty_filter)
			print('len(qty) is {} and len(qty_filter) is {}'.format(len(qty),len(qty_filter)))
			print(f"Correlations for quantity {i}: Spearman={s_corr}")
			corr_array_filter[counter] = s_corr
			counter = counter + 1

		### Also just plot spearman by itself
		fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.7)
		axs.bar(geo_quants, corr_array_filter[:])
		axs.set_ylim(-1,1)
		axs.set_xlabel('geo quantity', fontsize = label_size)
		axs.set_xticklabels(geo_quants, rotation=65)
		axs.set_ylabel('Spearman correlation', fontsize = label_size)		
		plt.suptitle('Correlation w/ $\\gamma / k_{{\\perp}}^2$ for KBM \n' + 'where $\\langle \\ldots \\rangle = \\int |\\phi|^2 \\ldots d\\theta / \\int |\\phi|^2 d\\theta$ (w/ g/krp2 filter {})'.format(mval), fontsize = 0.6*label_size)	
		# ~ plt.suptitle('Correlation with $\\gamma / k_{{\\perp}}^2$ for ' + instability_array[modetype_it] + ' modes, sample size = {}'.format(len(gamma_over_kperp2_nonzero_flat)) + ', where $\\langle \\ldots \\rangle = \\int |\\phi|^2 \\ldots d\\theta / \\int |\\phi|^2 d\\theta$')	
		# ~ plt.savefig(plot_out_dir+meta_scan_name+"_spearman_coefficients_mode_"+str(instability_array[modetype_it])+".pdf",bbox_inches='tight', pad_inches=0.1)
		# ~ plt.close()
		plt.show()

#### I prefer the gamma/kperp2 filter because each geometric quantity then has the same size.
'''
Further ideas: 

how these quantities vary across width height:

fit gamma/kperp2 to the different geometric quantities...

d gamma / d beta (we have this info!)

alpha_crit (harder, b/c we are not scanning for critical gradient)

add alpha and magnetic shear in here?

once this is done, we wish to compare all these quantities across squareness

'''

### Linear regression
mval = 5
quantities_filtered = []
counter = 0
for i, qty in enumerate(quantities, 1):
	# Filter qty
	gamma_over_kperp2_nonzero_flat_filter, qty_filter  = reject_outliers(np.array(gamma_over_kperp2_nonzero_flat), m = mval,filtertype='reduced_dual', correlated_array = np.array(qty))
	quantities_filtered.append(qty_filter)
# Stack the independent variables into a single matrix
X = np.vstack(quantities_filtered).T
# Add a column of ones to the independent variables matrix for the intercept term
X = np.hstack([np.ones((X.shape[0], 1)), X])
# Perform linear regression
coefficients, residuals, rank, s = np.linalg.lstsq(X, gamma_over_kperp2_nonzero_flat_filter, rcond=None)
# ~ print("Linear Regression Coefficients:", coefficients)

### Polynomial regression
# Create the design matrix for polynomial terms
def create_polynomial_features(X, degree):
	from itertools import combinations_with_replacement
	features = [np.ones(X.shape[0])]  # Add intercept term
	for deg in range(1, degree + 1):
		for items in combinations_with_replacement(range(X.shape[1]), deg):
			features.append(np.prod([X[:, item] for item in items], axis=0))
	return np.vstack(features).T

quantities_filtered = []
counter = 0
for i, qty in enumerate(quantities, 1):
	# Filter qty
	gamma_over_kperp2_nonzero_flat_filter, qty_filter  = reject_outliers(np.array(gamma_over_kperp2_nonzero_flat), m = mval,filtertype='reduced_dual', correlated_array = np.array(qty))
	quantities_filtered.append(np.array(qty_filter))
y = gamma_over_kperp2_nonzero_flat_filter
# Combine independent variables into a single matrix
X = np.vstack(quantities_filtered).T
# Ensure X is a NumPy array
X = np.array(X)
for degree in [1,2,3,4]:
	# Generate polynomial features
	X_poly = create_polynomial_features(X, degree)
	# Perform linear regression
	coefficients, residuals, rank, s = np.linalg.lstsq(X_poly, y, rcond=None)
	# ~ print("Quadratic Regression Coefficients:", coefficients)
	# Calculate R^2
	# Calculate predicted values
	y_pred = X_poly @ coefficients
	# Calculate R^2
	ss_res = np.sum((y - y_pred) ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r2 = 1 - (ss_res / ss_tot)
	print("R^2 for polynomial degree {}:".format(degree), r2)

# trying some fancier models
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

# Combine independent variables into a single matrix
X = np.vstack(quantities_filtered).T

# Define the degree of the polynomial for PolynomialFeatures
# ~ degree = 2
degree = 2

# Create a pipeline with standardization and SVR
model = make_pipeline(StandardScaler(),       # Step 1: Standardize the features
						PolynomialFeatures(degree),  # Step 2: Generate polynomial features
						SVR(kernel='poly', degree=degree, C=1.0, epsilon=0.1))  # Step 3: Fit SVR with polynomial kernel
# Fit the model
model.fit(X, y)
# Calculate R^2
r2 = model.score(X, y)
print("R^2:", r2)


from sklearn.ensemble import RandomForestRegressor
# Create a pipeline with standardization and a non-linear model
model = make_pipeline(StandardScaler(),       # Step 1: Standardize the features
						PolynomialFeatures(degree),  # Step 2: Generate polynomial features (optional, based on model needs)
						RandomForestRegressor())  # Step 3: Fit a non-linear regression model
# Fit the model
model.fit(X, y)
# Calculate R^2
r2 = model.score(X, y)
print("R^2 random forest:", r2)

#### finding the importance of each variable

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

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names_sorted, importances_sorted, color='b', align='center')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest Model')
plt.gca().invert_yaxis()
plt.show()

#### compute permute importance
from sklearn.inspection import permutation_importance

# Define the degree of the polynomial
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
important_idx = [i for i, name in enumerate(poly_feature_names) if name != '1']
important_features = [name for i, name in enumerate(poly_feature_names) if name != '1']

# Ensure the indices are within bounds
important_idx = [i for i in important_idx if i < len(result.importances_mean)]

# Get the permutation importance results for non-constant features
importances_mean = result.importances_mean[important_idx]
importances_std = result.importances_std[important_idx]

# Create a sorted list of features based on importance
sorted_idx = importances_mean.argsort()

# Plot the permutation importance
plt.figure(figsize=(10, 6))
plt.barh(np.array(important_features)[sorted_idx], importances_mean[sorted_idx], xerr=importances_std[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Permutation Importances in Random Forest Model")
plt.show()


#### Trying again with something directly from website: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def plot_permutation_importance(clf, X, y, ax):
	result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
	perm_sorted_idx = result.importances_mean.argsort()

	ax.boxplot(
		result.importances[perm_sorted_idx].T,
		vert=False,
		labels=X.columns[perm_sorted_idx],
	)
	ax.axvline(x=0, color="k", linestyle="--")
	return ax

# ~ X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train, y_train = X, y

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(trainingScores)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# ~ print(f"Baseline accuracy on test data: {clf.score(X_test, y_test):.2}")

mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mdi_importances.sort_values().plot.barh(ax=ax1)
ax1.set_xlabel("Gini importance")
plot_permutation_importance(clf, X_train, y_train, ax2)
ax2.set_xlabel("Decrease in accuracy score")
fig.suptitle(
	"Impurity-based vs. permutation importances on multicollinear features (train set)"
)
_ = fig.tight_layout()
'''


### Trying with the examples https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

# Define the degree of the polynomial
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

# Verify length of importances match the feature names minus constant term
# ~ assert len(result.importances_mean) == len(important_features), "Mismatch between importances and features"

# Get the permutation importance results
importances_mean = result.importances_mean
importances_std = result.importances_std

# Create a sorted list of features based on importance
sorted_idx = importances_mean.argsort()

# Plot the permutation importance
plt.figure(figsize=(10, 6))
sorted_important_features = np.array(geo_quants)[sorted_idx]
sorted_importances_mean = importances_mean[sorted_idx]
# ~ plt.barh(np.array(important_features)[sorted_idx], importances_mean[sorted_idx], xerr=importances_std[sorted_idx])
# ~ bars=plt.barh(np.linspace(-0.3,len(sorted_importances_mean)-0.7,len(sorted_importances_mean)), sorted_importances_mean, xerr=importances_std[sorted_idx])
bars=plt.barh(range(len(sorted_importances_mean)), len(sorted_importances_mean), sorted_importances_mean, xerr=importances_std[sorted_idx])
plt.yticks(range(len(importances_mean[sorted_idx])), sorted_important_features)

plt.xlabel("Permutation Importance")
plt.title("Permutation Importances in Random Forest Model")
plt.show()
# Debugging print statements to ensure all features are included
print(f"Original feature names: {poly_feature_names}")
print(f"Important feature names: {important_features}")
print(f"Sorted indices of importances: {sorted_idx}")
print(f"Feature importances (mean): {importances_mean}")
print(f"Feature importances (std): {importances_std}")


