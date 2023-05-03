import datetime
import pandas as pd
import numpy as np
from ctgan.synthesizers import CTGAN, TVAE

columns = [
    "age",
    "class of worker",
    "detailed industry recode",
    "detailed occupation recode",
    "education",
    "wage per hour",
    "enroll in edu inst last wk",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "hispanic origin",
    "sex",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "tax filer stat",
    "region of previous residence",
    "state of previous residence",
    "detailed household and family stat",
    "detailed household summary in household",
    "instance weight",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "live in this house 1 year ago",
    "migration prev res in sunbelt",
    "num persons worked for employer",
    "family members under 18",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "weeks worked in year",
    "year",
    "income"
]
binary_columns = [
    "sex",
    "year",
    "income",
]
cont_columns = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
    "instance weight",
]
multi_class_columns = list(set(columns) - set(binary_columns) - set(cont_columns))

data_fraction = 1.0
synthesizer = "tvae"
num_epochs_list = [30, 75, 100, 300]

for num_epochs in num_epochs_list:
    print(f"Training {synthesizer} for {num_epochs} epochs")
    census_df_train = pd.read_csv("/u/edithal/work/git_repos/csc2516_2023_project/dataset/census/census-income.data", header=None)
    census_df_train.columns = columns
    census_df_train= census_df_train.groupby("income").apply(lambda x: x.sample(frac=data_fraction))

    if synthesizer == "ctgan":
        model = CTGAN(epochs=num_epochs)
    elif synthesizer == "tvae":
        model = TVAE(epochs=num_epochs)
    else:
        raise Exception(f"Synthesizer {synthesizer} not defined!")
    model.fit(census_df_train, discrete_columns=(binary_columns + multi_class_columns))
    # Training fake data synthesizer for credit takes a lot of time, save the model to reuse it
    now = datetime.datetime.now()
    current_time = now.strftime("%d-%m-%Y-%H-%M-%S")
    model.save(f"/u/edithal/work/git_repos/csc2516_2023_project/models/census_{synthesizer}_{num_epochs}_epochs_{current_time}.pkl")