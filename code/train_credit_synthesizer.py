import time
import datetime
import pandas as pd
from ctgan.synthesizers import CTGAN, TVAE

# epochs = [30, 50, 75, 100, 200]
# synthesizers = ["ctgan", "tvae"]
credit_df = pd.read_csv("/u/edithal/work/git_repos/csc2516_2023_project/dataset/credit/creditcard.csv")
# Only use 25% of 280k rows which is 70k rows, otherwise transformation takes too much time (around 10 minutes)
# This is especially useful for hyperparameter tuning
# credit_df = credit_df.groupby("Class").apply(lambda x: x.sample(frac=0.25))

# for synthesizer in synthesizers:
#     for num_epoch in epochs:
        # print(f"Training {synthesizer} for {num_epoch} epochs")
        # if synthesizer == "ctgan":
        #     model = CTGAN(epochs=num_epoch)
        # elif synthesizer == "tvae":
        #     model = TVAE(epochs=num_epoch)
        # model.fit(credit_df, discrete_columns=["Class"])
        # now = datetime.datetime.now()
        # current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
        # model.save(
        #     f"/u/edithal/work/git_repos/csc2516_2023_project/models/credit_{synthesizer}_{num_epoch}_epochs_{current_time}.pkl"
        # )

num_epoch = 30
synthesizer = "tvae"
print(f"Training {synthesizer} for {num_epoch} epochs")
if synthesizer == "ctgan":
    model = CTGAN(epochs=num_epoch)
elif synthesizer == "tvae":
    model = TVAE(epochs=num_epoch)
model.fit(credit_df, discrete_columns=["Class"])
now = datetime.datetime.now()
current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
model.save(
    f"/u/edithal/work/git_repos/csc2516_2023_project/models/credit_{synthesizer}_{num_epoch}_epochs_{current_time}.pkl"
)