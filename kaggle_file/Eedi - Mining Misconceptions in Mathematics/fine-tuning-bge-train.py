EXP_NAME = "fine-tuning-bge"
DATA_PATH = "./eedi-mining-misconceptions-in-mathematics"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
COMPETITION_NAME = "eedi-mining-misconceptions-in-mathematics"
OUTPUT_PATH = "."
MODEL_OUTPUT_PATH = f"{OUTPUT_PATH}/trained_model"

RETRIEVE_NUM = 25

EPOCH = 2
LR = 2e-05
BS = 8
GRAD_ACC_STEP = 128 // BS
WEIGHT_DECAY = 0.01

TRAINING = True
DEBUG = False
WANDB = False


import datasets
import sentence_transformers
import os
import numpy as np

NUM_PROC = os.cpu_count()

from datasets import load_dataset, Dataset

import wandb
import polars as pl

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

if WANDB:
    # Settings -> add wandb api
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb.login(key=user_secrets.get_secret("wandbkey"))
    wandb.init(project=COMPETITION_NAME, name=EXP_NAME)
    REPORT_TO = "wandb"
else:
    REPORT_TO = "none"

REPORT_TO

train = pl.read_csv(f"{DATA_PATH}/train.csv")
misconception_mapping = pl.read_csv(f"{DATA_PATH}/misconception_mapping.csv")

common_col = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "QuestionText",
    "CorrectAnswer",
]

train_long = (
    train
    .select(
        pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]])
    )
    .unpivot(
        index=common_col,
        variable_name="AnswerType",
        value_name="AnswerText",
    )
    .with_columns(
        pl.concat_str(
            [
                pl.col("ConstructName"),
                pl.col("SubjectName"),
                pl.col("QuestionText"),
                pl.col("AnswerText"),
            ],
            separator=" ",
        ).alias("AllText"),
        pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
    )
    .with_columns(
        pl.concat_str(
            [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
        ).alias("QuestionId_Answer"),
    )
    .sort("QuestionId_Answer")
)

train_misconception_long = (
    train.select(
        pl.col(
            common_col + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]
        )
    )
    .unpivot(
        index=common_col,
        variable_name="MisconceptionType",
        value_name="MisconceptionId",
    )
    .with_columns(
        pl.col("MisconceptionType")
        .str.extract(r"Misconception([A-D])Id$")
        .alias("AnswerAlphabet"),
    )
    .with_columns(
        pl.concat_str(
            [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
        ).alias("QuestionId_Answer"),
    )
    .sort("QuestionId_Answer")
    .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
    .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
)
model = SentenceTransformer(MODEL_NAME)
train_long_vec = model.encode(
    train_long["AllText"].to_list(), normalize_embeddings=True
)
