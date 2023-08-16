import numpy as np
import pandas as pd
import pickle
import subprocess

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ae_gan import CTGANV2
from sklearn.model_selection import train_test_split

AE_TYPES = ['vanilla', 'denoising', 'vae', 'ee']

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
    dpath, npath = None, None
    cmd1, cmd2 = '', ''
    if dset in ["adult", "loan"]:
        cmd1 = f"find {BASE_DIR}/dataset/{dset} -name '*.data' -print"
        cmd2 = f"find {BASE_DIR}/dataset/{dset} -name '*.names' -print"
    elif dset == "credit":
        cmd1 = f"find {BASE_DIR}/dataset/{dset} -name '*.csv' -print"
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

def save_real_df(dset):
    if dset == 'adult':
        for t in ['data', 'test']:
            cmd = f"find {BASE_DIR}/dataset/adult -name '*.{t}' -print"
            dpath = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                ).stdout.decode('utf-8').strip()

            out = get_cols(dset)
            real_df = load_df(dpath,out['cols'][1])
            real_df["income_50k"] = real_df["income_50k"].str.rstrip('.')
            real_df.loc[real_df["income_50k"] == " <=50K"] = 0
            real_df.loc[real_df["income_50k"] == " >50K"] = 1
            real_df.to_csv(f"{BASE_DIR}/dataset/adult/real_{t}.csv", index=False)
            print(f"Real {t} dataframe for adult dataset saved: {BASE_DIR}/dataset/adult/real_{t}.csv")
    elif dset in ['credit', 'loan']:
        dt_path = f"{BASE_DIR}/dts/{dset}.dt"
        # load dt
        with open(dt_path,'rb') as f:
            dt = pickle.load(f)
        print("Data transformer loaded")

        # load td_train and td_val
        td_train_path = f"{BASE_DIR}/dataset/{dset}/{dset}_train.td"
        td_val_path = f"{BASE_DIR}/dataset/{dset}/{dset}_val.td"
        with open(td_train_path,'rb') as f:
            td_train = pickle.load(f)
        with open(td_val_path,'rb') as f:
            td_val = pickle.load(f)
        real_df = pd.concat([dt.inverse_transform(td_train),dt.inverse_transform(td_val)],axis=0)
        real_df.to_csv(f"{BASE_DIR}/dataset/{dset}/real_data.csv", index=False)
        print(f"Real train dataframe for {dset} dataset saved: {BASE_DIR}/dataset/{dset}/real_data.csv")

        # load td_test
        td_path = f"{BASE_DIR}/dataset/{dset}/{dset}_test.td"
        with open(td_path,'rb') as f:
            td_test = pickle.load(f)
        real_df = dt.inverse_transform(td_test)
        real_df.to_csv(f"{BASE_DIR}/dataset/{dset}/real_test.csv", index=False)
        print(f"Real test dataframe for {dset} dataset saved: {BASE_DIR}/dataset/{dset}/real_test.csv")

def save_fake_df(dset):
    for t in ['data', 'test']:
        real_df = pd.read_csv(f"{BASE_DIR}/dataset/{dset}/real_{t}.csv")
        # Sample all fake datasets
        for ae_type in AE_TYPES:
            cmd = f"find {BASE_DIR}/models/{dset}_tuned -name '*{ae_type}*' -print"
            mpath = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                ).stdout.decode('utf-8').strip()
            ae_gan = CTGANV2().load(mpath)
            fake_data = ae_gan.sample(real_df.shape[0])
            fake_data.to_csv(f"{BASE_DIR}/dataset/{dset}/fake_{t}_{ae_type}.csv", index=False) 
            print(f"Fake dataframe for {dset} ({ae_type}) dataset saved: {BASE_DIR}/dataset/{dset}/fake_{t}_{ae_type}.csv")

def ohe_cat(dset):
    if dset == "adult":
        cat_columns = get_cols("adult")['discrete_cols'][1]
        cat_columns.remove('income_50k')
        real_df = pd.read_csv(f"{BASE_DIR}/dataset/adult/real_test.csv")
        real_df = pd.get_dummies(real_df, columns=cat_columns)
        real_df.to_csv(f"{BASE_DIR}/dataset/adult/real_test.csv", index=False)
        for ae_type in AE_TYPES:
            fake_df = pd.read_csv(f"{BASE_DIR}/dataset/adult/fake_test_{ae_type}.csv")
            fake_df.loc[fake_df["income_50k"] == " <=50K"] = 0
            fake_df.loc[fake_df["income_50k"] == " >50K"] = 1
            fake_df = pd.get_dummies(fake_df, columns=cat_columns)
            col_union = np.union1d(fake_df.columns, real_df.columns)
            # Add missing columns
            for col_name in col_union:
                if col_name not in fake_df.columns:
                    fake_df[col_name] = 0
                if col_name not in real_df.columns:
                    real_df[col_name] = 0
            # Sort columns to ensure same ordering in fake and real data frames
            fake_df = fake_df.reindex(real_df.columns, axis=1)
            fake_df.to_csv(f"{BASE_DIR}/dataset/adult/fake_test_{ae_type}.csv", index=False)
    elif dset == "loan":
        real_df = pd.read_csv(f"{BASE_DIR}/dataset/loan/real_test.csv")
        real_df = pd.get_dummies(real_df, columns=['Family','Education'])
        real_df.to_csv(f"{BASE_DIR}/dataset/loan/real_test.csv", index=False)
        for ae_type in AE_TYPES:
            fake_df = pd.read_csv(f"{BASE_DIR}/dataset/loan/fake_test_{ae_type}.csv")
            fake_df = pd.get_dummies(fake_df, columns=['Family','Education'])
            col_union = np.union1d(fake_df.columns, real_df.columns)
            # Add missing columns
            for col_name in col_union:
                if col_name not in fake_df.columns:
                    fake_df[col_name] = 0
                if col_name not in real_df.columns:
                    real_df[col_name] = 0
            # Sort columns to ensure same ordering in fake and real data frames
            fake_df = fake_df.reindex(real_df.columns, axis=1)
            fake_df.to_csv(f"{BASE_DIR}/dataset/loan/fake_test_{ae_type}.csv", index=False)

def transform_adult():
    dt = DataTransformer()
    dpath, npath = get_paths("adult")
    print("Files paths fetched:")
    print(f"\t Data path: {dpath}")
    print(f"\t Column path: {npath}")
    
    out = get_cols("adult")

    df = load_df(dpath,out['cols'][1])
    print("Fitting the data transformer")
    dt.fit(df, discrete_columns=out['discrete_cols'][1])
    print("Successful!")

    savePath = f"{BASE_DIR}/dts/adult.dt"
    fh = open(savePath,'wb')
    pickle.dump(dt,fh)
    fh.close()
    print(f"Data transformer for adult saved at: {savePath}")

    # Split train data into train-val
    train_df, val_df = train_test_split(df,
                                        test_size=0.2, random_state=1, 
                                        shuffle=True, stratify=df["income_50k"])

    target_index = df.shape[1] - 1
    with open(f"{BASE_DIR}/dataset/adult/target.idx", 'w') as f:
        f.write(str(target_index))
    print(f"Target index stored at: {BASE_DIR}/dataset/adult/target.idx")

    print("Transforming the data")
    td_train = dt.transform(train_df)
    td_val = dt.transform(val_df)
    print("Data transformation completed")

    sp_train = f"{BASE_DIR}/dataset/adult/adult_train.td"
    sp_val = f"{BASE_DIR}/dataset/adult/adult_val.td"
    fh = open(sp_train,'wb')
    pickle.dump(td_train,fh)
    fh.close()
    fh = open(sp_val,'wb')
    pickle.dump(td_val,fh)
    fh.close()
    
    print(f"Transformed adult saved at: {sp_train}, {sp_val}")

def transform_credit():
    dt = DataTransformer()
    dpath, _ = get_paths("credit")
    print("Files paths fetched:")
    print(f"\t Data path: {dpath}")

    df = pd.read_csv(dpath)
    # Fetch 60k rows
    _, df = train_test_split(df,test_size=60000, random_state=5, shuffle=True, stratify=df["Class"])
    print("Fitting the data transformer")
    dt.fit(df, discrete_columns=("Class"))
    print("Successful!")

    savePath = f"{BASE_DIR}/dts/credit.dt"
    fh = open(savePath,'wb')
    pickle.dump(dt,fh)
    fh.close()
    print(f"Data transformer for credit saved at: {savePath}")

    # Split train data into train-test
    train_df, test_df = train_test_split(df, test_size=10000, random_state=5, shuffle=True, stratify=df["Class"])
    # Split train data further into train-val
    train_df, val_df = train_test_split(train_df, test_size=10000, random_state=5, shuffle=True, stratify=train_df["Class"])

    target_index = df.shape[1] - 1
    with open(f"{BASE_DIR}/dataset/credit/target.idx", 'w') as f:
        f.write(str(target_index))
    print(f"Target index stored at: {BASE_DIR}/dataset/credit/target.idx")

    print("Transforming the data")
    td_train = dt.transform(train_df)
    td_val = dt.transform(val_df)
    td_test = dt.transform(test_df)
    print("Data transformation completed")

    sp_train = f"{BASE_DIR}/dataset/credit/credit_train.td"
    sp_val = f"{BASE_DIR}/dataset/credit/credit_val.td"
    sp_test = f"{BASE_DIR}/dataset/credit/credit_test.td"
    fh = open(sp_train,'wb')
    pickle.dump(td_train,fh)
    fh.close()
    fh = open(sp_val,'wb')
    pickle.dump(td_val,fh)
    fh.close()
    fh = open(sp_test,'wb')
    pickle.dump(td_test,fh)
    fh.close()

    print(f"Transformed credit data saved at: {sp_train}, {sp_val}, {sp_test}")  

def transform_loan():
    dt = DataTransformer()
    dpath, npath = get_paths("loan")
    print("Files paths fetched:")
    print(f"\t Data path: {dpath}")
    print(f"\t Column path: {npath}")
    
    out = get_cols("loan")

    df = load_df(dpath,out['cols'][1])
    print("Fitting the data transformer")
    dt.fit(df, discrete_columns=out['discrete_cols'][1])
    print("Successful!")

    savePath = f"{BASE_DIR}/dts/loan.dt"
    fh = open(savePath,'wb')
    pickle.dump(dt,fh)
    fh.close()
    print(f"Data transformer for loan saved at: {savePath}")

    # Split train data into train-test
    train_df, test_df = train_test_split(df, test_size=1000, random_state=5, shuffle=True, stratify=df["Personal Loan"])
    # Split train data further into train-val
    train_df, val_df = train_test_split(train_df, test_size=1000, random_state=5, shuffle=True, stratify=train_df["Personal Loan"])

    target_index = df.columns.get_loc('Personal Loan')
    with open(f"{BASE_DIR}/dataset/loan/target.idx", 'w') as f:
        f.write(str(target_index))
    print(f"Target index stored at: {BASE_DIR}/dataset/loan/target.idx")

    print("Transforming the data")
    td_train = dt.transform(train_df)
    td_val = dt.transform(val_df)
    td_test = dt.transform(test_df)
    print("Data transformation completed")

    sp_train = f"{BASE_DIR}/dataset/loan/loan_train.td"
    sp_val = f"{BASE_DIR}/dataset/loan/loan_val.td"
    sp_test = f"{BASE_DIR}/dataset/loan/loan_test.td"
    fh = open(sp_train,'wb')
    pickle.dump(td_train,fh)
    fh.close()
    fh = open(sp_val,'wb')
    pickle.dump(td_val,fh)
    fh.close()
    fh = open(sp_test,'wb')
    pickle.dump(td_test,fh)
    fh.close()

    print(f"Transformed loan data saved at: {sp_train}, {sp_val}, {sp_test}") 

def write_transform_to_file(dset):
    if dset == 'adult':
        transform_adult()
    elif dset == 'credit':
        transform_credit()
    elif dset == 'loan':
        transform_loan()

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

def sample_fake_credit(dt,td,model):
    # Get fake and real dataframes
    fake_df = model.sample(td['train'].shape[0]+td['val'].shape[0])
    val_df = dt.inverse_transform(td['val'])

    # Get real feature and target values
    val_y = val_df.pop("Class").to_numpy()
    val_y = np.eye(2)[val_y]
    val_X = val_df.to_numpy()

    # Get fake feature and target values
    fake_y = fake_df.pop("Class").to_numpy()
    fake_y = np.eye(2)[fake_y]
    fake_X = fake_df.to_numpy()
    fake_train_X, fake_val_X, fake_train_y, fake_val_y = train_test_split(fake_X, fake_y,
                                                                            test_size=10000, random_state=1,
                                                                            shuffle=True, stratify=fake_y.argmax(-1))
    
    return [(fake_train_X,fake_train_y,fake_val_X,fake_val_y),(val_X,val_y)]

def sample_fake_loan(dt,td,model):
    # Get fake and real dataframes
    fake_df = model.sample(td['train'].shape[0]+td['val'].shape[0])
    val_df = dt.inverse_transform(td['val'])

    # Get OHE categorical columns
    fake_df = pd.get_dummies(fake_df, columns=['Family','Education'])
    val_df = pd.get_dummies(val_df, columns=['Family','Education'])
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
    val_y = val_df.pop("Personal Loan").to_numpy()
    val_y = np.eye(2)[val_y]
    val_X = val_df.to_numpy()

    # Get fake feature and target values
    fake_y = fake_df.pop("Personal Loan").to_numpy()
    fake_y = np.eye(2)[fake_y]
    fake_X = fake_df.to_numpy()
    fake_train_X, fake_val_X, fake_train_y, fake_val_y = train_test_split(fake_X, fake_y,
                                                                            test_size=1000, random_state=1,
                                                                            shuffle=True, stratify=fake_y.argmax(-1))
    
    return [(fake_train_X,fake_train_y,fake_val_X,fake_val_y),(val_X,val_y)]