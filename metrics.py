import numpy as np
import pandas as pd

from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def alpha_beta_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    metric = AlphaPrecision()

    real_loader = GenericDataLoader(real_df)
    synth_loader = GenericDataLoader(synth_df)

    score = metric.evaluate(
        X_gt=real_loader,
        X_syn=synth_loader,
    )

    return score


def audit_alpha_precision(real_df, synth_df, alpha):
    metric = AlphaPrecision()

    real_loader = GenericDataLoader(real_df)
    synth_loader = GenericDataLoader(synth_df)

    return metric.audit_alpha_precision(real_loader, synth_loader, alpha)




def train_synth_test_real(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    train_classes = y_train.nunique()

    if train_classes < 2:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "tstr_accuracy": float(accuracy_score(y_test, y_pred)),
        "tstr_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "tstr_macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "tstr_train_classes": int(train_classes),
    }