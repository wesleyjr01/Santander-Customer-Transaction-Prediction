import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier
import pickle
import datetime

train = pd.read_pickle("train.pkl").drop("ID_code", axis=1)
test = pd.read_pickle("test.pkl")
test_id = test['ID_code'].values
test.drop(['ID_code'], axis=1, inplace=True)

result = np.zeros(test.shape[0])  # Vector to be filled
oof = np.zeros(len(train))  # Vector to be filled
seed = 17
list_of_splits = [2, 3]
n_repeats = 1
for splits in list_of_splits:
  rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=n_repeats, random_state=seed)
  for counter, (train_index, valid_index) in enumerate(rskf.split(train, train.target), 1):
    print(counter)

    # Train data
    X_train = train.drop(['target'], axis=1).iloc[train_index]
    y_train = train.target.iloc[train_index]

    # Validation data
    X_valid = train.drop(['target'], axis=1).iloc[valid_index]
    y_valid = train.target.iloc[valid_index]

    # Training
    model = CatBoostClassifier(iterations=100000,
                               learning_rate=0.017,
                               l2_leaf_reg=40,
                               bootstrap_type='Bernoulli',
                               depth=3,
                               subsample=0.63,
                               eval_metric='AUC',
                               random_seed=seed,
                               # bagging_temperature = 0.2,
                               # od_type='Iter',
                               # metric_period = 75
                               )
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),
              cat_features=None,
              use_best_model=True,
              verbose=True)

    # Output .pkl model
    model_name = "CatBoost_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.pkl'
    model_name = model_name.replace(':', '_')
    pickle.dump(model, open(model_name, 'wb'))

    # Feed OOF Vector with Val Prediction
    oof[valid_index] = model.predict(X_valid)

    # Feed Test Prediction Vector
    result += model.predict(test)

n_models = sum([i * n_repeats for i in list_of_splits])
print(f'n_models:{n_models}')
submission = pd.DataFrame({"ID_code": test_id})
submission['target'] = result / n_models
filename = "CatBoost__submission" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv"
filename = filename.replace(':', '_')
submission.to_csv(filename, index=False)
