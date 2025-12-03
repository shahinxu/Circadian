from sklearn.decomposition import PCA

def create_eigengenes(expression_scaled, n_components=50):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(expression_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    return components, pca, explained_variance