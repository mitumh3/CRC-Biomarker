from concurrent.futures import ThreadPoolExecutor
# Method for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Feature Selection
import itertools
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Ignore wanring
import warnings
warnings.filterwarnings("ignore")

# Basic process
import numpy as np
from tqdm import tqdm

# Method for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def select_features(data, num_features, test_size, method):
    y = data["target"]
    X = data.drop(["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    # Transform
    if method == "Standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    # Training
    X_train = scaler.fit_transform(X_train)
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(X_train, y_train)
    cols = selector.get_support(indices=True)
    cols = X.iloc[:, cols].columns
    return cols

# Making combination inputs
def combinations(features, combinations):
    features = list(itertools.combinations(features, combinations))
    return features

# Model name
def model_name(selected_features):
    return '-'.join(selected_features)

def filter_iqr(df, threshold):
    iqr_values = df.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25), axis=1)
    # Filter genes with IQR > 0.5
    keep_gene_lst = df.index[iqr_values > threshold]
    return keep_gene_lst

def prepare_y_dfs(validation):
    """Efficiently prepare y_dfs array."""
    dt = np.dtype([('DFS', np.bool_), ('DFSM', np.float64)])
    y_dfs_list = [
        (validation.loc[idx, 'DFS'], validation.loc[idx, 'DFSM']) 
        for idx in validation.index
    ]
    return np.array(y_dfs_list, dtype=dt)

def process_feature_set(feature_set, train_test, validation, test_size, method, num_folds, y_dfs, path):
    """Process a single feature set."""
    features = feature_set.split("-")

    # Prepare datasets
    data1 = train_test[features + ["target"]]
    data2 = validation[features + ["target"]]

    y = data1["target"]
    X = data1.drop(["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Select scaler
    scaler = StandardScaler() if method == "Standard" else MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Validation data
    X_valid = data2.drop(["target"], axis=1)
    X_valid = scaler.transform(X_valid)
    y_valid = data2["target"]
    y_stage = validation["stage"]

    return [
        X_train, X_test, X_valid, y_train, 
        y_test, y_valid, num_folds, feature_set, 
        y_stage, y_dfs, path
    ]

def input_prepare(train_test, validation, num_folds, selected_features, test_size, method, path):
    """Main function to prepare input data in parallel."""
    y_dfs = prepare_y_dfs(validation)

    # Parallel processing of features
    models = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_feature_set, feature_set, train_test, validation, 
                test_size, method, num_folds, y_dfs, path
            )
            for feature_set in selected_features
        ]
        for future in tqdm(futures, desc="Processing features"):
            models.append(future.result())
    
    return models