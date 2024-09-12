#!/usr/bin/env python

"""Example for training a random forest classifier in sklearn
   and using mlflow to save a model.
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import time
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


def wait_model_transition(model_name, model_version, stage):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name,
                                                         version=model_version,
                                                         )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            client.transition_model_version_stage(
              name=model_name,
              version=model_version,
              stage=stage,
            )
            break
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modelPath', type=str,
                        help='Name of mlflow artifact path location to drop model.')
    parser.add_argument('--outputTestData', type=str,
                        help='Name of output csv file if writing split test data.')
    args = parser.parse_args()

    model_path = args.modelPath

    # Load a standard machine learning dataset
    cancer = load_breast_cancer()

    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']

    # Optionally write out a subset of the data, used in this tutorial for inference with the API
    if args.outputTestData:
        train, test = train_test_split(df, test_size=0.2)
        del test['target']
        test.to_csv('test.csv', index=False)

        features = [x for x in list(train.columns) if x != 'target']
        x_raw = train[features]
        y_raw = train['target']
    else:
        features = [x for x in list(df.columns) if x != 'target']
        x_raw = df[features]
        y_raw = df['target']

    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                        test_size=.20,
                                                        random_state=123,
                                                        stratify=y_raw)

    # Build a classifier sklearn pipeline
    clf = RandomForestClassifier(n_estimators=100,
                                 min_samples_leaf=2,
                                 class_weight='balanced',
                                 random_state=123)

    preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('randomforestclassifier', clf)])

    # Train the model
    model.fit(x_train, y_train)

    def overwrite_predict(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return [round(x, 4) for x in result[:, 1]]
        return wrapper

    # Overwriting the model to use predict to output probabilities
    model.predict = overwrite_predict(model.predict_proba)

    try:
        mlflow.sklearn.save_model(model, model_path)
        print("Generating new model in path {}".format(model_path))

    except:
        print("Using existing model in path {}".format(model_path))
        pass


if __name__ == "__main__":
    main()
