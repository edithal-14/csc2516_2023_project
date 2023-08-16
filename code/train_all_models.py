import pickle
from ctgan.synthesizers.ae_gan import CTGANV2, AutoEncoderType
from aegan_utils import IntrusionDataset, load_covtype

ROOT_DIR = "/u/edithal/work/git_repos/csc2516_2023_project/"

dataloader = IntrusionDataset(path=f"{ROOT_DIR}dataset/intrusion/")
intrusion_train_df, intrusion_valid_df, intrusion_test_df, intrusion_discrete_columns = dataloader.load_intrusion()

with open(f"{ROOT_DIR}dataset/intrusion/data_transformer.pkl", "rb") as fp:
    intrusion_dt = pickle.load(fp)
with open(f"{ROOT_DIR}dataset/intrusion/transformed_data.pkl", "rb") as fp:
    intrusion_trans = pickle.load(fp)

covtype_train_df, covtype_valid_df, covtype_test_df, covtype_discrete_columns = load_covtype(path=f"{ROOT_DIR}dataset/covtype/")

with open(f"{ROOT_DIR}dataset/covtype/data_transformer.pkl", "rb") as fp:
    covtype_dt = pickle.load(fp)
with open(f"{ROOT_DIR}dataset/covtype/transformed_data.pkl", "rb") as fp:
    covtype_trans = pickle.load(fp)


# idx = 0
# n_runs = 2
# dataset = ["intrusion", "covtype"]
# param_dict = {
#     "ae_type": [AutoEncoderType.VANILLA, AutoEncoderType.VANILLA],
#     "ae_dim": [(256, 128), (256, 128)],
#     "embedding_dim": [64, 64],
#     "generator_dim": [(256, 256), (256, 256, 256, 256)],
#     "discriminator_dim": [(256, 256), (256, 256, 256, 256)],
#     "discriminator_steps": [1, 5],
#     "pac": [4, 16],
#     "batch_size": [512, 256],
#     "ae_batch_size": [512, 256],
#     "epochs": [300, 300],
#     "ae_epochs": [100, 100], 
#     "autoencoder_lr": [1e-3, 2e-4],
#     "generator_lr": [2e-4, 1e-5],
#     "discriminator_lr": [2e-4, 1e-5],
# }

# idx = 2
# n_runs = 2
# dataset = ["intrusion", "covtype"]
# param_dict = {
#     "ae_type": [AutoEncoderType.DENOISING, AutoEncoderType.DENOISING],
#     "ae_dim": [(256, 128), (256, 128)],
#     "embedding_dim": [64, 64],
#     "generator_dim": [(256, 256), (256, 256)],
#     "discriminator_dim": [(256, 256), (256, 256)],
#     "discriminator_steps": [1, 5],
#     "pac": [16, 8],
#     "batch_size": [256, 256],
#     "ae_batch_size": [256, 256],
#     "epochs": [300, 300],
#     "ae_epochs": [100, 100], 
#     "autoencoder_lr": [1e-4, 1e-3],
#     "generator_lr": [2e-4, 1e-5],
#     "discriminator_lr": [2e-4, 1e-5],
# }

idx = 4
# n_runs = 2
# dataset = ["intrusion", "covtype"]
# param_dict = {
#     "ae_type": [AutoEncoderType.ENTITY, AutoEncoderType.ENTITY],
#     "ae_dim": [(256, 128), (256, 128)],
#     "embedding_dim": [64, 128],
#     "generator_dim": [(256, 256, 256, 256), (256, 256)],
#     "discriminator_dim": [(256, 256, 256, 256), (256, 256)],
#     "discriminator_steps": [1, 5],
#     "pac": [4, 16],
#     "batch_size": [256, 256],
#     "ae_batch_size": [256, 256],
#     "epochs": [300, 300],
#     "ae_epochs": [100, 100], 
#     "autoencoder_lr": [1e-3, 1e-4],
#     "generator_lr": [2e-4, 2e-4],
#     "discriminator_lr": [2e-4, 2e-4],
# }

idx = 6
n_runs = 2
dataset = ["intrusion", "covtype"]
param_dict = {
    "ae_type": [AutoEncoderType.VARIATIONAL, AutoEncoderType.VARIATIONAL],
    "ae_dim": [(256, 128, 64, 32), (256, 128, 64)],
    "embedding_dim": [64, 128],
    "generator_dim": [(256, 256), (256, 256)],
    "discriminator_dim": [(256, 256), (256, 256)],
    "discriminator_steps": [5, 1],
    "pac": [4, 8],
    "batch_size": [256, 256],
    "ae_batch_size": [256, 256],
    "epochs": [300, 300],
    "ae_epochs": [100, 100], 
    "autoencoder_lr": [2e-4, 1e-3],
    "generator_lr": [2e-4, 2e-4],
    "discriminator_lr": [2e-4, 2e-4],
}

for i in range(n_runs):
    print(f"Starting with model number: {idx+1}")
    inner_param_dict = dict()
    for key in param_dict:
        inner_param_dict[key] = param_dict[key][i]

    model = CTGANV2(
        **inner_param_dict,
    )

    if dataset[i] == "intrusion":
        data_trans = intrusion_trans
        discrete_columns = intrusion_discrete_columns
        target_index = intrusion_train_df.shape[1]- 1
        dt = intrusion_dt
    elif dataset[i] == "covtype":
        data_trans = covtype_trans 
        discrete_columns = covtype_discrete_columns
        target_index = covtype_train_df.shape[1]- 1
        dt = covtype_dt
    else:
        raise Exception("Invalid dataset")

    model.fit(data_trans, discrete_columns=discrete_columns, dt=dt, is_pre_transformed=True, target_index=target_index)

    ae_type = param_dict["ae_type"][i].value
    dataset_type = dataset[i]

    model.save(
        f"{ROOT_DIR}models/tuned_{dataset_type}_{ae_type}.pkl"
    )

    idx+=1