import numpy as np
import pandas as pd
import pickle
import subprocess

from ctgan.data_transformer import DataTransformer
from sklearn.model_selection import train_test_split

BASE_DIR = subprocess.run(
    "git rev-parse --show-toplevel".split(),
    stdout=subprocess.PIPE
    ).stdout.decode('utf-8').strip()

def load_df(path: str, columns: list=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = columns
    return df

def read_target_idx(dset):
    with open(f"{BASE_DIR}/dataset/{dset}/target.idx", 'r') as f:
        target_index = int(f.read())
    return target_index

def get_paths(dset):
    cmd1 = f"find {BASE_DIR}/dataset/{dset} -name '*.data' -print"
    cmd2 = f"find {BASE_DIR}/dataset/{dset} -name '*.names' -print"
    dpath = subprocess.run(
        cmd1,
        shell=True,
        stdout=subprocess.PIPE,
        ).stdout.decode('utf-8').strip()
    npath = subprocess.run(
        cmd2,
        shell=True,
        stdout=subprocess.PIPE
        ).stdout.decode('utf-8').strip()
    return dpath, npath

def get_cols(dset, print_cols=False):
    _, npath = get_paths(dset)
    out = {'cols': ["awk -v RS='\n\n' 'END {print}' " + npath +  " | awk -F: '{print $1}'", list()],
           'discrete_cols': ["awk -v RS='\n\n' 'END {print}' " + npath +  " | grep -v continuous | awk -F: '{print $1}'", list()]}
    for key, val in out.items():
        val[1] = subprocess.run(
            val[0],
            shell=True, 
            stdout=subprocess.PIPE
            ).stdout.decode('utf-8').strip().split('\n')
    if print_cols:
        print(f"\t Columns: {out['cols'][1]}")
        print(f"\t Discrete Columns: {out['discrete_cols'][1]}")
    return out

def write_transform_to_file(dset):
    dt = DataTransformer()
    dpath, npath = get_paths(dset)
    print("Files paths fetched:")
    print(f"\t Data path: {dpath}")
    print(f"\t Column path: {npath}")
    
    out = get_cols(dset)

    df = load_df(dpath,out['cols'][1])
    print("Fitting the data transformer")
    dt.fit(df, discrete_columns=out['discrete_cols'][1])
    print("Successful!")

    savePath = f"{BASE_DIR}/dts/{dset}.dt"
    fh = open(savePath,'wb')
    pickle.dump(dt,fh)
    fh.close()
    print(f"Data transformer for {dset} saved at: {savePath}")

    # Split train data into train-val
    train_df, val_df = train_test_split(df,
                                        test_size=0.2, random_state=1, 
                                        shuffle=True, stratify=df["income_50k"])

    target_index = df.shape[1] - 1
    with open(f"{BASE_DIR}/dataset/{dset}/target.idx", 'w') as f:
        f.write(str(target_index))
    print(f"Targe index stored at: {BASE_DIR}/dataset/{dset}/target.idx")

    print("Transforming the data")
    td_train = dt.transform(train_df)
    td_val = dt.transform(val_df)
    print("Data transformation completed")

    sp_train = f"{BASE_DIR}/dataset/{dset}/{dset}_train.td"
    sp_val = f"{BASE_DIR}/dataset/{dset}/{dset}_val.td"
    fh = open(sp_train,'wb')
    pickle.dump(td_train,fh)
    fh.close()
    fh = open(sp_val,'wb')
    pickle.dump(td_val,fh)
    fh.close()
    
    print(f"Transformed {dset} saved at: {sp_train}, {sp_val}")

def load_transform_from_file(dset):
    dt_path = f"{BASE_DIR}/dts/{dset}.dt"
    # load dt
    fh = open(dt_path,'rb')
    dt = pickle.load(fh)
    fh.close() 
    print("Data transformer loaded")

    # load td_train
    td = {}
    for t in ['train', 'val']:
        td_path = f"{BASE_DIR}/dataset/{dset}/{dset}_{t}.td"
        fh = open(td_path,'rb')
        td[t] = pickle.load(fh)
        fh.close()

    print("Transformed data loaded")
    return dt, td

def sample_fake_adult(dt,td,model):
    # Get fake and real dataframes
    fake_df = model.sample(td['train'].shape[0]+td['val'].shape[0])
    val_df = dt.inverse_transform(td['val'])

    # Get OHE categorical columns
    cat_columns = get_cols('adult')['discrete_cols'][1]
    fake_df = pd.get_dummies(fake_df, cat_columns)
    val_df = pd.get_dummies(val_df, cat_columns)
    col_union = np.union1d(fake_df.columns, val_df.columns)
    # Add missing columns
    for col_name in col_union:
        if col_name not in fake_df.columns:
            fake_df[col_name] = 0
        if col_name not in val_df.columns:
            val_df[col_name] = 0
    # Sort columns to ensure same ordering in fake and real data frames
    fake_df = fake_df.reindex(val_df.columns, axis=1)

    # Get real feature and target values
    val_y = val_df[["income_50k_ >50K", "income_50k_ <=50K"]].to_numpy()
    val_X = val_df.drop(["income_50k_ >50K", "income_50k_ <=50K"], axis=1).to_numpy()

    # Get fake feature and target values
    fake_y = fake_df[["income_50k_ >50K", "income_50k_ <=50K"]].to_numpy()
    fake_X = fake_df.drop(["income_50k_ >50K", "income_50k_ <=50K"], axis=1).to_numpy()
    fake_train_X, fake_val_X, fake_train_y, fake_val_y = train_test_split(fake_X, fake_y,
                                                                            test_size=0.2, random_state=1, 
                                                                            shuffle=True, stratify=fake_y.argmax(-1))
    
    return [(fake_train_X,fake_train_y,fake_val_X,fake_val_y),(val_X,val_y)]