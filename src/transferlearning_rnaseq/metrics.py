from SFDA.metrics import LEEP, LogME
import numpy as np
from sklearn.decomposition import PCA
import pdb
def compute_gbc(source_features, target_labels, pca_dim=0.8, covariance_type='diagonal'):
    # Apply PCA dimensionality reduction
    pca = PCA(n_components=pca_dim)
    reduced_features = pca.fit_transform(source_features)
    
    # Compute per-class statistics
    unique_classes = np.unique(target_labels)
    class_means = {}
    class_covs = {}
    
    for c in unique_classes:
        class_features = reduced_features[(target_labels == c).flatten()]
        class_means[c] = np.mean(class_features, axis=0)
        
        if covariance_type == 'diagonal':
            class_covs[c] = np.diag(np.var(class_features, axis=0))
        elif covariance_type == 'spherical':
            class_covs[c] = np.eye(pca_dim) * np.mean(np.var(class_features, axis=0))
        else:
            class_covs[c] = np.cov(class_features.T)
    
    # Compute pairwise Bhattacharyya coefficients
    gbc_score = 0
    for i in unique_classes:
        for j in unique_classes:
            if i != j:
                mu_i, mu_j = class_means[i], class_means[j]
                cov_i, cov_j = class_covs[i], class_covs[j]
                
                # Compute Bhattacharyya distance
                cov = (cov_i + cov_j) / 2
                db = 0.125 * np.dot(np.dot((mu_i - mu_j).T, np.linalg.inv(cov)), (mu_i - mu_j)) + \
                     0.5 * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(cov_i) * np.linalg.det(cov_j)))
                
                # Convert distance to coefficient
                bc = np.exp(-db)
                gbc_score -= bc 
    
    return gbc_score

import numpy as np

def coding_rate(Z, eps=1E-4):
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1/(n*eps) * Z.transpose() @ Z))
    return 0.5 * rate

def trans_rate(Z, y, eps=1E-4):
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)
    for i in range(K):
        RZY += coding_rate(Z[(y == i).flatten()], eps)
    return RZ - RZY / K

def compute_metrics(features, y):
    logme = LogME()
    return{"LogME": logme.fit(features, y), "LEEP": LEEP(features, y), "GBC": compute_gbc(features, y), "transrate":trans_rate(features, y)}

#def compute_metrics(features, y):
    #logme = LogME()
    #return{"LogME": 0, "NLEEP": 0, "GBC": compute_gbc(features, y), "transrate":trans_rate(features, y)}
