import pickle
from ctgan.synthesizers.ae_gan import CTGANV2, AutoEncoderType
from aegan_utils import IntrusionDataset, train_model
from sklearn.model_selection import train_test_split
import os

ROOT_DIR = "/u/edithal/work/git_repos/csc2516_2023_project/"

# columns = [
#     "duration",
#     "protocol_type",
#     "service",
#     "flag",
#     "src_bytes",
#     "dst_bytes",
#     "land",
#     "wrong_fragment",
#     "urgent",
#     "hot",
#     "num_failed_logins",
#     "logged_in",
#     "num_compromised",
#     "root_shell",
#     "su_attempted",
#     "num_root",
#     "num_file_creations",
#     "num_shells",
#     "num_access_files",
#     "num_outbound_cmds",
#     "is_host_login",
#     "is_guest_login",
#     "count",
#     "srv_count",
#     "serror_rate",
#     "srv_serror_rate",
#     "rerror_rate",
#     "srv_rerror_rate",
#     "same_srv_rate",
#     "diff_srv_rate",
#     "srv_diff_host_rate",
#     "dst_host_count",
#     "dst_host_srv_count",
#     "dst_host_same_srv_rate",
#     "dst_host_diff_srv_rate",
#     "dst_host_same_src_port_rate",
#     "dst_host_srv_diff_host_rate",
#     "dst_host_serror_rate",
#     "dst_host_srv_serror_rate",
#     "dst_host_rerror_rate",
#     "dst_host_srv_rerror_rate",
#     "class"
# ]
# multi_class_columns = [
#     "wrong_fragment",
#     "urgent",
#     "hot",
#     "num_failed_logins",
#     "num_compromised",
#     "su_attempted",
#     "num_root",
#     "num_file_creations",
#     "num_shells",
#     "num_access_files",
#     "protocol_type",
#     "service",
#     "flag",
# ]
# binary_columns = ["land", "logged_in", "root_shell", "is_guest_login"]
# intrusion_df = pd.read_csv("/u/edithal/work/git_repos/csc2516_2023_project/dataset/intrusion/kddcup.data_10_percent", header=None)
# intrusion_df.columns = columns
# # Training TVAE on whole dataset produces sigma containing NaN
# data_fraction = 1.0
# # Only use a fraction of dataset using stratified sampling to maintain class samples ratio
# # This is especially useful for hyperparameter tuning of epochs
# intrusion_df = intrusion_df.groupby("class").apply(lambda x: x.sample(frac=data_fraction))
# # remove columns: num_outbound_cmds, is_host_login since they contain the same value and provide no extra information
# intrusion_df = intrusion_df.drop(columns=["num_outbound_cmds", "is_host_login"])
# for col in binary_columns:
#     intrusion_df[col] = intrusion_df[col].astype(np.int8)

# synthesizer = "tvae"
# num_epochs_list = [30, 75, 100, 300]

# for num_epochs in num_epochs_list:
#     print(f"Training {synthesizer} for {num_epochs} epochs")
#     if synthesizer == "ctgan":
#         model = CTGAN(epochs=num_epochs)
#     elif synthesizer == "tvae":
#         model = TVAE(epochs=num_epochs)
#     else:
#         raise Exception(f"Synthesizer {synthesizer} not defined!")

#     model.fit(intrusion_df, discrete_columns=(binary_columns + multi_class_columns + ["class"]))
#     # Training fake data synthesizer for credit takes a lot of time, save the model to reuse it
#     now = datetime.datetime.now()
#     current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
#     model.save(f"/u/edithal/work/git_repos/csc2516_2023_project/models/intrusion_{synthesizer}_{num_epochs}_epochs_{current_time}.pkl")

#####################
# TRAINING
#####################

dataloader = IntrusionDataset(path=f"{ROOT_DIR}dataset/intrusion/")
intrusion_train_df, intrusion_valid_df, intrusion_test_df, intrusion_discrete_columns = dataloader.load_intrusion()

with open(f"{ROOT_DIR}dataset/intrusion/data_transformer.pkl", "rb") as fp:
    dt = pickle.load(fp)
with open(f"{ROOT_DIR}dataset/intrusion/transformed_data.pkl", "rb") as fp:
    intrusion_trans = pickle.load(fp)

idx = 37
n_runs = 12
param_dict = {
    "ae_type": [AutoEncoderType.VARIATIONAL for i in range(n_runs)],
    "ae_dim": [(256, 128, 64, 32) for i in range(n_runs)],
    "embedding_dim": [64 for i in range(n_runs)],
    "generator_dim": [(256, 256) for i in range(n_runs)],
    "discriminator_dim": [(256, 256) for i in range(n_runs)],
    "discriminator_steps": [5 for i in range(n_runs)],
    "pac": [4 for i in range(n_runs)],
    "batch_size": [512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256],
    "ae_batch_size": [512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256],
    "epochs": [100, 50, 100, 50, 100, 50, 100, 50, 100, 50, 100, 50],
    "ae_epochs": [100, 50, 100, 50, 100, 50, 100, 50, 100, 50, 100, 50],
    "autoencoder_lr": [1e-4, 1e-4, 1e-4, 1e-4, 2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    "generator_lr": [1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4],
    "discriminator_lr": [1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4],
}


for i in range(n_runs):
    print(f"Starting with model number: {idx}")
    inner_param_dict = dict()
    for key in param_dict:
        inner_param_dict[key] = param_dict[key][i]

    model = CTGANV2(
        **inner_param_dict,
    )

    model.fit(intrusion_trans, discrete_columns=intrusion_discrete_columns, dt=dt, is_pre_transformed=True, target_index=intrusion_train_df.shape[1]- 1)

    model.save(
        f"{ROOT_DIR}models/intrusion_ae_gan_{idx}.pkl"
    )

    idx+=1

##################
# INFERENCE
##################

# intrusion_valid_X, intrusion_valid_y = dataloader.get_Xy(intrusion_valid_df)

# start = 12
# n_models = 12

# scores = []

# for i in range(start, start+n_models):
#     print(f"Processing {i+1} model\n")
#     model_file = f"{ROOT_DIR}models/intrusion_ae_gan_{i+1}.pkl"
#     if not os.path.exists(model_file):
#         print.write(f"Skipping {i+1} model\n")
#         continue
#     model = CTGANV2.load(model_file)
#     # Sample fake train and validation data
#     idx = 0
#     while idx < 5:
#         try:
#             intrusion_fake_df = model.sample(intrusion_train_df.shape[0] + intrusion_valid_df.shape[0])
#             intrusion_fake_X, intrusion_fake_y = dataloader.get_Xy(intrusion_fake_df)
#             intrusion_fake_train_X, intrusion_fake_valid_X, intrusion_fake_train_y, intrusion_fake_valid_y = train_test_split(
#                 intrusion_fake_X, intrusion_fake_y, test_size=intrusion_valid_df.shape[0], random_state=1, shuffle=True, stratify=intrusion_fake_y.argmax(-1)
#             )
#             break
#         except:
#             idx+=1
#             print("Invalid fake data generated, trying again...\n")

#     if idx >= 5:
#         scores.append(-100)
#         print(f"Failed to generate fake data after {idx} tries, skipping...")
#         continue

#     best_test_score = -float("inf")
#     for i in range(5):
#         test_score = train_model(
#             intrusion_fake_train_X,
#             intrusion_fake_train_y,
#             intrusion_fake_valid_X,
#             intrusion_fake_valid_y,
#             intrusion_valid_X,
#             intrusion_valid_y,
#             input_dim=215,
#             output_dim=23,
#             batch_size=2048,
#             num_epochs=300,
#             model_type="classification",
#             show_print_training_score=False,
#             verbose=False,
#             scorer_type="f1_macro",
#         )
#         if test_score > best_test_score:
#             best_test_score = test_score
#     print(f"Test score: {best_test_score}\n")
#     scores.append(best_test_score)
# print("Scores\n")
# print("\n".join([str(score) for score in scores]))