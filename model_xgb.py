# TODO: dimensional reduction
# TODO: tuning parameter
# TODO:

import numpy as np
import pandas as pd
from scipy import sparse
import pickle
from sklearn.preprocessing import (Imputer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, make_scorer
import xgboost as xgb

# True if not yet do preprocessing
preprocessing = False

def preprocess_data():
    """
    Load raw data and do preprocessing
    """
    train_data = pickle.load(open("train_data.pkl", "rb"))
    test_data = pickle.load(open("test_data.pkl", "rb"))

    # cat_features = ["HasName", "Intact", "Sex", "IsMix", "SimpleColor"]
    cat_features = ["HasName", "Intact", "Sex", "Year"]
    num_features = ["Hour", "Weekday", "Month", "AgeinDays"]
    label_name = "OutcomeType"

    # fill missing data
    imr = Imputer()
    train_data["AgeinDays"] = imr.fit_transform(
        train_data["AgeinDays"].reshape(-1, 1))
    test_data["AgeinDays"] = imr.transform(
        test_data["AgeinDays"].reshape(-1, 1))

    # try binary data with NaN rows !not work
    # train_data["Intact"] = train_data["Intact"].apply(
    #     lambda x: 0 if x == "Unknown" else x)
    # train_data["Intact"] = train_data["Intact"].apply(
    #     lambda x: 1 if x == "Yes" else -1)
    # test_data["Intact"] = test_data["Intact"].apply(
    #     lambda x: 0 if x == "Unknown" else x)
    # test_data["Intact"] = test_data["Intact"].apply(
    #     lambda x: 1 if x == "Yes" else -1)
    # train_data["Sex"] = train_data["Sex"].apply(
    #     lambda x: 0 if x == "Unknown" else x)
    # train_data["Sex"] = train_data["Sex"].apply(
    #     lambda x: 1 if x == "Male" else -1)
    # test_data["Sex"] = test_data["Sex"].apply(
    #     lambda x: 0 if x == "Unknown" else x)
    # test_data["Sex"] = test_data["Sex"].apply(
    #     lambda x: 1 if x == "Female" else -1)
    # train_bin = train_data[bin_features].values
    # test_bin = test_data[bin_features].values

    # convert categorical features to numerical
    for feature in cat_features:
        le = LabelEncoder()
        train_data[feature] = le.fit_transform(train_data[feature].values)
        test_data[feature] = le.transform(test_data[feature].values)

    # convert categorical label to numerical
    le = LabelEncoder()
    train_data[label_name] = le.fit_transform(train_data[label_name].values)
    label_list = list(le.inverse_transform([0, 1, 2, 3, 4]))

    # onehot encode for categorical features
    ohe = OneHotEncoder()
    train_cat = ohe.fit_transform(train_data[cat_features])
    test_cat = ohe.transform(test_data[cat_features])

    # scale numerical features
    stdsc = StandardScaler()
    train_num = stdsc.fit_transform(train_data[num_features])
    test_num = stdsc.transform(test_data[num_features])

    # vectorize the text features, no need to do scaler or idf
    train_data["BreedColor"] = train_data["Breed"] + " " + train_data["Color"]
    test_data["BreedColor"] = test_data["Breed"] + " " + test_data["Color"]
    # tfv = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2),
    #                       token_pattern=r"\w{1,}", min_df=3,
    #                       norm="l2", use_idf=True, smooth_idf=True,
    #                       sublinear_tf=True)
    tfv = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2),
                          token_pattern=r"\w{1,}", min_df=3,
                          norm=None, use_idf=False, smooth_idf=False,
                          sublinear_tf=False)
    train_text = tfv.fit_transform(train_data["BreedColor"])
    test_text = tfv.transform(test_data["BreedColor"])

    # return the final data
    train = sparse.hstack((train_num, train_cat, train_text))
    label = train_data[label_name]
    test = sparse.hstack((test_num, test_cat, test_text))

    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump(train, f, protocol=-1)
        pickle.dump(label, f, protocol=-1)
        pickle.dump(test, f, protocol=-1)
        pickle.dump(label_list, f, protocol=-1)


def load_data():
    """
    Load preprocessed data
    """
    if preprocessing:
        preprocess_data()
    with open("preprocessed_data.pkl", "rb") as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        label_list = pickle.load(f)
    return train_data, train_label, test_data, label_list


def XGB_test(train_data, train_label, test_data, label_list):
    """
    Construct a test xgboost model and create a submission file
    """
    clf = xgb.XGBClassifier(n_estimators=224, max_depth=8, min_child_weight=6,
                            gamma=1,
                            objective="multi:softprob",
                            subsample=0.8, colsample_bytree=0.8)
    clf.fit(train_data, train_label, eval_metric="mlogloss")
    test_predict = clf.predict_proba(test_data)
    print("There is {0} features".format(clf._features_count))
    print(clf.feature_importances_)

    # create submission file
    sample = pd.read_csv("input/sample_submission.csv")
    sample[label_list] = test_predict
    sample.to_csv("submit/xgb_test.csv", index=False)
    # 329/1604 logloss 0.73917


def XGB_cv(train_data, train_label):
    """
    Do cross validation
    """
    clf = xgb.XGBClassifier(n_estimators=113, max_depth=7, objective="multi:softprob",
                            subsample=0.8, colsample_bytree=0.8)
    kfold = StratifiedKFold(train_label, n_folds=2)
    scores = []
    for k, (train, test) in enumerate(kfold):
        clf.fit(train_data.tocsc()[train], train_label[train],
                eval_set=[(train_data.tocsc()[train], train_label[train]),
                          (train_data.tocsc()[test], train_label[test])],
                eval_metric="mlogloss", verbose=True)
        scores.append("{0} time: {1}".format(k+1, clf.evals_result()))

    print(scores)


def XGB_para_tuning(train_data, train_label):
    """
    Do parameter tuning
    """
    clf = xgb.XGBClassifier(objective="multi:softprob")
    # param_grid = [{"max_depth": [3,5,7,9,12,15,17,25],
    #                "learning_rate": [0.01,0.015,0.025,0.05,0.1],
    #                "gamma": [0.05,0.1,0.3,0.5,0.7,0.9,1],
    #                "min_child_weight": [1,3,5,7],
    #                "subsample": [0.6,0.7,0.8,0.9,1],
    #                "colsample_bytree": [0.6,0.7,0.8,0.9,1],
    #                "reg_alpha": [0,0.01,0.1,1],
    #                "reg_lambda": [0,0.1,0.5,1]}]
    param_grid = [{"max_depth": [3,5],
                   "learning_rate": [0.01,0.005],
                   "gamma": [0.05],
                   "min_child_weight": [1],
                   "subsample": [0.6],
                   "colsample_bytree": [0.6],
                   "reg_alpha": [0],
                   "reg_lambda": [0]}]
    scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    gs = GridSearchCV(clf, param_grid, fit_params={"eval_metric":"mlogloss"},
                      scoring=scorer, cv=3)
    gs.fit(train_data, train_label)
    print(gs.best_params_, gs.best_score_)

def XGB_default(train_data, train_label, test_data, label_list):
    """
    Construct a default xgboost model and create a submission file
    """
    clf = xgb.XGBClassifier(objective="multi:softprob")
    clf.fit(train_data, train_label, eval_metric="mlogloss")
    test_predict = clf.predict_proba(test_data)
    print("There is {0} features".format(clf._features_count))
    print(clf.feature_importances_)

    # create submission file
    sample = pd.read_csv("input/sample_submission.csv")
    sample[label_list] = test_predict
    sample.to_csv("submit/xgb_default.csv", index=False)


def RF_default(train_data, train_label, test_data, label_list):
    """
    Construct a default random forest model and create a submission file
    """
    clf = RandomForestClassifier(n_estimators=120)
    clf.fit(train_data, train_label)
    test_predict = clf.predict_proba(test_data)

    # create submission file
    sample = pd.read_csv("input/sample_submission.csv")
    sample[label_list] = test_predict
    sample.to_csv("submit/rf_default.csv", index=False)


def ET_default(train_data, train_label, test_data, label_list):
    """
    Construct a default extra tree model and create a submission file
    """
    clf = ExtraTreesClassifier(n_estimators=120)
    clf.fit(train_data, train_label)
    test_predict = clf.predict_proba(test_data)

    # create submission file
    sample = pd.read_csv("input/sample_submission.csv")
    sample[label_list] = test_predict
    sample.to_csv("submit/et_default.csv", index=False)


def main():
    train_data, train_label, test_data, label_list = load_data()
    # XGB_default(train_data, train_label, test_data, label_list)
    XGB_test(train_data, train_label, test_data, label_list)
    # !not work
    # RF_default(train_data, train_label, test_data, label_list)
    # ET_default(train_data, train_label, test_data, label_list)


if __name__ == "__main__":
    main()
