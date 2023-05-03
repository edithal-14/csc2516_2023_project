import datetime
import pandas as pd
from ctgan.synthesizers import CTGAN, TVAE
from ctgan.synthesizers.ae_gan import CTGANV2, AutoEncoderType
import pickle

from aegan_utils import load_news, get_news_Xy

ROOT_DIR = "/u/edithal/work/git_repos/csc2516_2023_project/"

news_train_df, news_valid_df, news_test_df, news_discrete_columns = load_news(path=f"{ROOT_DIR}dataset/news/")

# cat_columns = [
#     "data_channel_is_lifestyle",
#     "data_channel_is_entertainment",
#     "data_channel_is_bus",
#     "data_channel_is_socmed",
#     "data_channel_is_tech",
#     "data_channel_is_world",
#     "weekday_is_monday",
#     "weekday_is_tuesday",
#     "weekday_is_wednesday",
#     "weekday_is_thursday",
#     "weekday_is_friday",
#     "weekday_is_saturday",
#     "weekday_is_sunday",
#     "is_weekend",
# ]
# news_df = pd.read_csv(f"{ROOT_DIR}dataset/news/OnlineNewsPopularity.csv")
# # Remove leading whitespace from column names
# news_df.columns = [col_name.lstrip() for col_name in news_df.columns.to_list()]
# # Remove url column
# news_df = news_df.drop(columns=["url"])

# synthesizer = "tvae"
# num_epochs_list = [30, 50, 75, 100, 300]

# for num_epochs in num_epochs_list:
#     print(f"Training {synthesizer} for {num_epochs} epochs")
#     if synthesizer == "ctgan":
#         model = CTGAN(epochs=num_epochs)
#     elif synthesizer == "tvae":
#         model = TVAE(epochs=num_epochs)
#     else:
#         raise Exception(f"Synthesizer {synthesizer} not defined!")

#     model.fit(news_df, discrete_columns=cat_columns)
#     now = datetime.datetime.now()
#     current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
#     model.save(
#         f"/u/edithal/work/git_repos/csc2516_2023_project/models/news_{synthesizer}_{num_epochs}_epochs_{current_time}.pkl"
#     )

# news_trans = model.transform(news_train_df, discrete_columns=news_discrete_columns)

# Save data transformer and transformed data to a file
# with open(f"{ROOT_DIR}dataset/news/data_transformer.pkl", "wb") as fp:
#     pickle.dump(model._transformer, fp)
# with open(f"{ROOT_DIR}dataset/news/transformed_data.pkl", "wb") as fp:
#     pickle.dump(news_trans, fp)

# # Load data transformer and transformed data
with open(f"{ROOT_DIR}/dataset/news/data_transformer.pkl", "rb") as fp:
    dt = pickle.load(fp)
with open(f"{ROOT_DIR}dataset/news/transformed_data.pkl", "rb") as fp:
    news_trans = pickle.load(fp)

# idx = 1
# n_runs = 4
# param_dict = {
#     "autoencoder_lr": [1e-4, 1e-4, 1e-4, 1e-4],
#     "generator_lr": [1e-5, 1e-5, 2e-4, 2e-4],
#     "discriminator_lr": [1e-5, 1e-5, 2e-4, 2e-4],
#     "batch_size": [512, 256, 512, 256],
#     "ae_batch_size": [512, 256, 512, 256],
#     "epochs": [300, 150, 300, 150],
#     "ae_epochs": [100, 50, 100, 50],
# }

idx = 25
n_runs = 12
param_dict = {
    "ae_type": [AutoEncoderType.VARIATIONAL for i in range(n_runs)],
    "ae_dim": [(256, 128) for i in range(n_runs)],
    "embedding_dim": [128 for i in range(n_runs)],
    "generator_dim": [(256, 256) for i in range(n_runs)],
    "discriminator_steps": [5 for i in range(n_runs)],
    "pac": [8 for i in range(n_runs)],
    "autoencoder_lr": [1e-4, 1e-4, 1e-4, 1e-4, 2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
    "generator_lr": [1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4],
    "discriminator_lr": [1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4, 1e-5, 1e-5, 2e-4, 2e-4],
    "batch_size": [512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256],
    "ae_batch_size": [512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256],
    "epochs": [300, 150, 300, 150, 300, 150, 300, 150, 300, 150, 300, 150],
    "ae_epochs": [100, 50, 100, 50, 100, 50, 100, 50, 100, 50, 100, 50],
}


for i in range(n_runs):
    inner_param_dict = dict()
    for key in param_dict:
        inner_param_dict[key] = param_dict[key][i]

    model = CTGANV2(
        **inner_param_dict,
    )

    model.fit(news_trans, discrete_columns=news_discrete_columns, dt=dt, is_pre_transformed=True)

    now = datetime.datetime.now()
    current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
    model.save(
        f"{ROOT_DIR}models/news_ae_gan_{idx}.pkl"
    )

    idx+=1