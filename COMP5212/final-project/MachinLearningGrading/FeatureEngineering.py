import pandas as pd

def polyFeature(X:pd.DataFrame, degree:int)->pd.DataFrame:
    """
    Generate polynomial features
    :param X: input data
    :param degree: degree of polynomial
    :return: polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    #return the dataframe
    X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
    return X_poly

def feaScaling(X:pd.DataFrame)->pd.DataFrame:
    """
    Feature scaling
    :param X: input data
    :return: scaled data
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def feaInteract(X:pd.DataFrame)->pd.DataFrame:
    """
    Generate interaction features
    :param X: input data
    :return: interaction features
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(interaction_only=True,include_bias=False)
    X_interact = poly.fit_transform(X)
    #return the dataframe
    X_interact = pd.DataFrame(X_interact, columns=poly.get_feature_names_out(X.columns))
    return X_interact

def feaCluster(X:pd.DataFrame,n_clusters:int=3)->pd.DataFrame:
    """
    Generate cluster features
    :param X: input data
    :param n_clusters: number of clusters
    :return: cluster features
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    X["cluster"] = kmeans.fit_predict(X)
    return X


def feabinning(X:pd.DataFrame):
    for column in X.columns:
        X[f"{column}_bin"] = pd.cut(X[column], bins=3, labels=False)

    return X
    