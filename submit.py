import numpy as np

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def make_sparse(weights):
    arr = weights
    arr = -np.sort(-arr, axis=0)
    
    a=arr[512]
    for i in range  (0, len(weights)):
        if( weights[i] < a ):
            weights[i]=0

    return weights 

################################
# Non Editable Region Starting #
################################
def my_fit( X_trn, y_trn ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# Youe method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding
	
    # Compute the least squares solution
    wt = np.linalg.lstsq(X_trn, y_trn, rcond=None)[0]
    # simple least square solution result
    wt= make_sparse(wt)
   
    return wt
# Result 
# 2.7463984639999977 207.72659650217582 149.96858356221836 0.25390625
				# Return the trained model
## Below code is for Coordinate Gradiant Descent algo
## It takes approx 10 min to train the model 
## Its accuracy and error is not good 

#def my_fit( X_trn, y_trn ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# Youe method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding
#  input=X_trn 
 # output=y_trn 
#  (samp, feat) = input.shape
#  coeff = np.zeros(feat) # Initialize coefficients with zeros
#  old_coeff = np.zeros(feat)
#  alpha = 0.1
#  max_iter = 1000
#  tol = 1e-4
#  for _ in range(max_iter):
    # Perform coordinate descent
#    for j in range(samp):
#      X_j = input[j,:]
#      theta_j = coeff
#      z = np.dot(X_j, coeff) - theta_j * X_j
#      p = 1 / (1 + np.exp(-z))
#      r = output[j] - p
#      c = np.dot(X_j ** 2, p * (1 - p))
#      if c == 0:
#        coeff[j] = 0
#      else:
#        rho = np.dot(X_j, r) / c
#        if rho < -alpha:
#          coeff[j] = (rho + alpha)/c
#        elif rho > alpha:
#          coeff[j] = (rho - alpha)/c
#        else:
#          coeff[j] = 0
  # Check convergence
#    if np.linalg.norm(coeff - old_coeff) < tol:
#     break
#    old_coeff = coeff.copy()
#  return coeff				# Return the trained model

# def my_fit(X_trn, y_trn):
#     num_samples, num_features = X_trn.shape
#     n_iter=100
#     tolerance=1e-06
#     # Normalize the input data
    # X_normalized = (X_trn - np.mean(X_trn, axis=0)) / np.std(X_trn, axis=0)

    # # Initialize weights as zeros
    # weights = np.zeros(num_features)

    # # Learning rate
    # learn_rate = 0.0002
    # # Perform coordinate gradient descent
    # for _ in range(n_iter):
    #     prev_weights = np.copy(weights)

    #     for j in range(num_features):
    #         # Compute the coordinate-wise gradient
    #         gradient_j = 2 * np.dot(X_normalized[:, j], X_normalized[:, j] * weights[j] - y_trn)

    #         # Update the j-th coordinate of weights
    #         weights[j] -= learn_rate * gradient_j

    #     # Check convergence using L2-norm of weight updates
    #     weight_updates = np.linalg.norm(weights - prev_weights)
    #     if weight_updates < tolerance:
    #         break

    # return weights
    #### RESULTS OF THIS COORDINATE DESCENT CODE 
##### 1.5245459562636154 1.108909189392236e+31 5.677615049688251e+30 0.3368875

