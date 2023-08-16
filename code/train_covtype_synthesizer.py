import pickle
from ctgan.synthesizers.ae_gan import CTGANV2, AutoEncoderType

from aegan_utils import load_covtype

ROOT_DIR = "/u/edithal/work/git_repos/csc2516_2023_project/"

# covtype_df = pd.read_csv(f"/u/edithal/work/git_repos/csc2516_2023_project/dataset/covtype/covtype.data", header=None)
# covtype_df.columns = [str(i) for i in np.arange(covtype_df.shape[1])]
# discrete_columns = [str(i) for i in np.arange(10, covtype_df.shape[1])]

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
#     model.fit(covtype_df, discrete_columns=discrete_columns)
#     now = datetime.datetime.now()
#     current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
#     model.save(f"/u/edithal/work/git_repos/csc2516_2023_project/models/covtype_{synthesizer}_{num_epochs}_epochs_{current_time}.pkl")

covtype_train_df, covtype_valid_df, covtype_test_df, covtype_discrete_columns = load_covtype(path=f"{ROOT_DIR}dataset/covtype/")

# model = CTGANV2()

# covtype_trans = model.transform(covtype_train_df, discrete_columns=covtype_discrete_columns)

# Save data transformer and transformed data to a file
# with open(f"{ROOT_DIR}dataset/covtype/data_transformer.pkl", "wb") as fp:
#     pickle.dump(model._transformer, fp)
# with open(f"{ROOT_DIR}dataset/covtype/transformed_data.pkl", "wb") as fp:
#     pickle.dump(covtype_trans, fp)

# # Load data transformer and transformed data
with open(f"{ROOT_DIR}dataset/covtype/data_transformer.pkl", "rb") as fp:
    dt = pickle.load(fp)
with open(f"{ROOT_DIR}dataset/covtype/transformed_data.pkl", "rb") as fp:
    covtype_trans = pickle.load(fp)

idx = 37
n_runs = 12
param_dict = {
    "ae_type": [AutoEncoderType.VARIATIONAL for i in range(n_runs)],
    "ae_dim": [(256, 128, 64) for i in range(n_runs)],
    "embedding_dim": [128 for i in range(n_runs)],
    "generator_dim": [(256, 256) for i in range(n_runs)],
    "discriminator_dim": [(256, 256) for i in range(n_runs)],
    "discriminator_steps": [1 for i in range(n_runs)],
    "pac": [8 for i in range(n_runs)],
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

    model.fit(covtype_trans, discrete_columns=covtype_discrete_columns, dt=dt, is_pre_transformed=True, target_index=covtype_train_df.shape[1]- 1)

    model.save(
        f"{ROOT_DIR}models/covtype_ae_gan_{idx}.pkl"
    )

    idx+=1