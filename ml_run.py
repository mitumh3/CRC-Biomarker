from ml_func import parameter_tuning
from preprocess_func import input_prepare
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from multiprocessing import cpu_count
from tqdm import tqdm

def worker_function(value):
    """Worker function to process each value."""
    return parameter_tuning(value)

def parallel(values, num_workers=-1):
    """Parallel execution using ProcessPoolExecutor with a progress bar."""
    max_workers = num_workers if num_workers > 0 else cpu_count()
    print(f"Using {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to wrap the results generator and display the progress bar
        results = list(tqdm(executor.map(worker_function, values), total=len(values), desc="Building..."))


def get_data(clin_path, genes_path, target_col, extra_cols=[]):
    clin= pd.read_csv(
        clin_path,
        sep="\t",
        index_col="!Sample_geo_accession"
    )
    clin["target"] = [1 if i == " colorectal cancer" else 0 for i in clin[target_col]]
    clin = clin[["target"] + extra_cols]

    genes= pd.read_csv(
        genes_path, 
        sep="\t",
        index_col="ID_REF"
    )
    data = pd.merge(clin, genes.T, left_index=True, right_index=True)
    return data

def batch_features(feature_list, batch_size=1000):
    for i in range(0, len(feature_list), batch_size):
        yield feature_list[i:i + batch_size]

# Machine Learning Process
if __name__ == '__main__':
    with open("featurelist.csv", "r") as file:
        content = file.read()
        features = content.split(",")
    with open("used_featurelist.csv", "r") as file:
        content = file.read()
        used_features = content.split(",")
    features = list(set(features) - set(used_features))
    print("No. of features: ", len(features))

    train_data = get_data(
        clin_path="data/GSE164191_clinicaldata.txt",
        genes_path="data/GSE164191_genes.txt",
        target_col="status"
    )
    train_data.dropna(inplace=True)

    val_data = get_data(
        clin_path="data/GSE39582_clinicaldata.txt",
        genes_path="data/GSE39582_genes.txt",
        target_col="status",
        extra_cols=["stage", "DFS", "DFSM"]
    )
    val_data.dropna(inplace=True)

    batches = batch_features(features, batch_size=10000)
    batch_no = 1
    for batch in batches:
        print("Start batch no. ", batch_no)
        input_wraps = input_prepare(
            train_test = train_data, 
            validation = val_data,  
            num_folds=5, 
            selected_features = batch, 
            test_size=0.5, 
            method = "Standard",
            path = "output"
        )
        
        # Execute parallel processing
        print(f"Available CPU cores: {cpu_count()}")
        parallel(input_wraps, num_workers=-1)

        batch_no += 1
        with open("used_featurelist.csv", "w") as file:
            file.write(",".join(used_features + batch))
    print("Run completed")
    #TODO: basicly, y is the same in every cases. It is duplicated 1333500 times in input wraps.