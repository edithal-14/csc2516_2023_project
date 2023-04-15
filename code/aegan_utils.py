import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score

def load_news(path="../dataset/news/"):
    df = pd.read_csv(f"{path}/OnlineNewsPopularity.csv")
    # Remove leading whitespace from column names
    df.columns = [col_name.lstrip() for col_name in df.columns.to_list()]
    # Set categorical columns
    cat_columns = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
        "is_weekend",
    ]
    # Remove url column
    df = df.drop(columns=["url"])
    # Split out test dataset
    train_df, test_df = train_test_split(df, test_size=8000, random_state=1, shuffle=True)
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=1, shuffle=True)
    # Do not perform one hot encoding since the categorical columns are already binary valued
    return train_df, valid_df, test_df, cat_columns


def get_news_Xy(df):
    y = df.pop("shares").to_numpy()[..., None]
    X = df.to_numpy()
    return X, y


def load_covtype(path="../dataset/covtype/"):
    covtype_df = pd.read_csv(f"{path}covtype.data", header=None)
    covtype_df.columns = [str(i) for i in np.arange(covtype_df.shape[1])]
    discrete_columns = [str(i) for i in np.arange(10, covtype_df.shape[1])]
    # Split out test dataset
    # Random sample 55000 rows due to computational limitation
    _, covtype_df = train_test_split(covtype_df, test_size=55000, random_state=5, shuffle=True, stratify=covtype_df.iloc[:,-1])
    # Test 5k
    train_df, test_df = train_test_split(covtype_df, test_size=5000, random_state=1, shuffle=True, stratify=covtype_df.iloc[:,-1])
    # Valid 5k, Train 50k
    train_df, valid_df = train_test_split(train_df, test_size=5000, random_state=3, shuffle=True, stratify=train_df.iloc[:,-1])
    return train_df, valid_df, test_df, discrete_columns


def get_covtype_Xy(df):
    data = df.to_numpy()
    num_classes = 7
    # classes are from 1 to 7 so subtract by one
    y = np.eye(num_classes)[data[:,-1] - 1]
    X = data[:,:-1]
    return X, y

class IntrusionDataset:
    def __init__(self):
        super(IntrusionDataset, self).__init__()
        self.columns = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "class"
        ]
        self.multi_class_columns = [
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "num_compromised",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "protocol_type",
            "service",
            "flag",
        ]
        self.binary_columns = ["land", "logged_in", "root_shell", "is_guest_login"]
        self.target_column = ["class"]

    def load_intrusion(self, path="../dataset/intrusion/"):
        intrusion_df = pd.read_csv(f"{path}kddcup.data_10_percent", header=None)
        intrusion_df.columns = self.columns
        # remove columns: num_outbound_cmds, is_host_login since they contain the same value and provide no extra information
        intrusion_df = intrusion_df.drop(columns=["num_outbound_cmds", "is_host_login"])
        for col in self.binary_columns:
            intrusion_df[col] = intrusion_df[col].astype(np.int8)
        
        # Separate rare classes to facilitate stratified sampling
        class_counts = intrusion_df["class"].value_counts()
        rare_classes = class_counts[class_counts < 100].keys().to_list()
        intrusion_df = intrusion_df[~intrusion_df["class"].isin(rare_classes)]
        
        discrete_columns = self.binary_columns + self.multi_class_columns + self.target_column

        # Split out test dataset
        # Random sample 55000 rows due to computational limitation
        _, train_df = train_test_split(intrusion_df, test_size=55000, random_state=5, shuffle=True, stratify=intrusion_df["class"])
        # Test 5k
        train_df, test_df = train_test_split(train_df, test_size=5000, random_state=1, shuffle=True, stratify=train_df["class"])
        # Valid 5k, Train 50k
        train_df, valid_df = train_test_split(train_df, test_size=5000, random_state=3, shuffle=True, stratify=train_df["class"])

        # Add rare classes if at least 3 rows for that rare class is present
        for class_name in rare_classes:
            rare_df = intrusion_df[intrusion_df["class"] == class_name]
            idx = rare_df.shape[0] // 3
            if idx > 0:
                test_df = pd.concat([test_df, rare_df.iloc[:idx]])
                valid_df = pd.concat([valid_df, rare_df.iloc[idx:2*idx]])
                train_df = pd.concat([train_df, rare_df.iloc[2*idx:]])

        # Add missing categorical values to train_df
        for col in discrete_columns:
            missing_categories = list(set(intrusion_df[col].unique()) - set(train_df[col].unique()))
            for category in missing_categories:
                train_df = pd.concat([train_df, intrusion_df[intrusion_df[col] == category].head(1)])


        return train_df, valid_df, test_df, discrete_columns


class BenchmarkMLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim=100) -> None:
        super(BenchmarkMLP, self).__init__()
        num_layers = 1
        layer_dim = (hidden_dim,)
        self.layers = []
        dim = input_dim
        for i in range(num_layers):
            self.layers.append(
                nn.Linear(dim, layer_dim[i])
            )
            self.layers.append(nn.ReLU())
            dim = layer_dim[i]
        self.layers.append(
            nn.Linear(dim, output_dim)
        )
        self.layers = nn.ModuleList(self.layers)
        self.layers.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def scorer(output, target, scorer_type, model_type, f1_class):
    if scorer_type == "f1_macro" and model_type == "classification":
        return f1_score(target.argmax(-1), output.argmax(-1), average="macro")
    elif scorer_type == "f1_micro" and model_type == "classification":
        return f1_score(target.argmax(-1), output.argmax(-1), average="micro")
    elif scorer_type == "accuracy" and model_type == "classification":
        correct = (target.argmax(-1) == output.argmax(-1)).sum()
        total = target.shape[0]
        return correct/total
    elif scorer_type in ("f1_minority", "f1_majority") and model_type == "classification" and f1_class is not None:
        return f1_score(target.argmax(-1), output.argmax(-1), average="binary", pos_label=f1_class)
    elif scorer_type == "r2" and model_type == "regression":
        return r2_score(target, output)
    else:
        raise Exception(f"Invalid scorer type: {scorer_type} and model type: {model_type} combination")

def train_model(
    train_X,
    train_y,
    valid_X,
    valid_y,
    test_X,
    test_y,
    input_dim=784,
    output_dim=10,
    num_epochs=10,
    batch_size=64,
    device="cuda",
    model_type="classification",
    verbose=True,
    use_best_validation=True,
    hidden_dim=100,
    show_print_training_score=False,
    scorer_type="f1_macro"
):
    # scorer_type supported: f1_macro, f1_micro, f1_minority, f1_majority, r2, accuracy
    # model_type supported: regression, classification
    train_len = train_X.shape[0]
    num_batches_per_epoch = train_len // batch_size
    if train_len % batch_size != 0:
        num_batches_per_epoch += 1
    if model_type == "classification":
        critetion = nn.CrossEntropyLoss()
        unique_elements, counts = np.unique(train_y.argmax(-1), return_counts=True)
        if scorer_type == "f1_minority":
            f1_class = unique_elements[np.argmin(counts)]
        elif scorer_type == "f1_majority":
            f1_class = unique_elements[np.argmax(counts)]
        else:
            f1_class = None
    elif model_type == "regression":
        critetion = nn.MSELoss()
        f1_class = None
    else:
        raise Exception("Invalid model_type argument")

    model = BenchmarkMLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())

    train_X = torch.Tensor(train_X).to(device)
    train_y = torch.Tensor(train_y).to(device)
    valid_X = torch.Tensor(valid_X).to(device)
    valid_y = torch.Tensor(valid_y).to(device)
    test_X = torch.Tensor(test_X).to(device)
    test_y = torch.Tensor(test_y).to(device)

    best_valid_score = -float("inf")
    best_model_weights = model.state_dict()
    for epoch in tqdm(range(num_epochs)):
        model.train()
        idx = 0
        epoch_loss = 0
        train_score = 0
        # Shuffle train_X and train_y before every epoch
        random_ids = np.random.permutation(train_len)
        train_X = train_X[random_ids]
        train_y = train_y[random_ids]
        for _ in range(num_batches_per_epoch):
            input = train_X[idx : idx + batch_size]
            targets = train_y[idx : idx + batch_size]
            idx += batch_size
            output = model(input)
            loss = critetion(output, targets)
            if show_print_training_score:
                train_score += scorer(output.cpu().detach(), targets.cpu().detach(), scorer_type, model_type, f1_class)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if model_type == "classification":
            epoch_loss = epoch_loss / train_len
        elif model_type == "regression":
            epoch_loss = epoch_loss / num_batches_per_epoch
        else:
            raise Exception("Invalid model_type argument")
        train_score = train_score / num_batches_per_epoch

        model.eval()
        with torch.no_grad():
            output = model(valid_X)
            if model_type == "classification":
                valid_loss = critetion(output, valid_y) / valid_y.shape[0]
            elif model_type == "regression":
                valid_loss = critetion(output, valid_y)
            else:
                raise Exception("Invalid model_type argument")
            valid_score = scorer(output.cpu().detach(), valid_y.cpu().detach(), scorer_type, model_type, f1_class)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_model_weights = model.state_dict()

        if verbose or epoch % 25 == 0:
            if show_print_training_score:
                print(f"Epoch: {epoch + 1}, Training Loss: {epoch_loss:.4f}, Training Score: {train_score:.4f}, Valid Loss: {valid_loss:.4f}, Valid score: {valid_score:.4f}")
            else:
                print(f"Epoch: {epoch + 1}, Training Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid score: {valid_score:.4f}")

    print(f"Best validation score: {best_valid_score}")
    if use_best_validation:
        model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        output = model(test_X)
        test_score = scorer(output.cpu().detach(), test_y.cpu().detach(), scorer_type, model_type, f1_class)
    
    return test_score