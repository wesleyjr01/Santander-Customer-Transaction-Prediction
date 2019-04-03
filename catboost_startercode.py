import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier
import pickle
import datetime

def augment(train, num_n=1, num_p=2):
    newtrain = [train]

    n = train[train.target == 0]
    for i in range(num_n):
        newtrain.append(n.apply(lambda x: x.values.take(np.random.permutation(len(n)))))

    for i in range(num_p):
        p = train[train.target > 0]
        newtrain.append(p.apply(lambda x: x.values.take(np.random.permutation(len(p)))))
    return pd.concat(newtrain)

train = pd.read_pickle("train.pkl").drop("ID_code", axis=1)
test = pd.read_pickle("test.pkl")
test_id = test['ID_code'].values
test.drop(['ID_code'], axis=1, inplace=True)

result = np.zeros(test.shape[0])  # Vector to be filled
oof = np.zeros(len(train))  # Vector to be filled
seed = 17
list_of_splits = [3, 5, 7]
n_repeats = 3
for splits in list_of_splits:
  rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=n_repeats, random_state=seed)
  for counter, (train_index, valid_index) in enumerate(rskf.split(train, train.target), 1):
    print(counter)

    # Train data
    X_train = train.iloc[train_index]
    X_train = augment(X_train)
    y_train = X_train.target
    X_train = X_train.drop(['target'], axis=1)


    # Validation data
    X_valid = train.drop(['target'], axis=1).iloc[valid_index]
    y_valid = train.target.iloc[valid_index]

    # Training
    model = CatBoostClassifier(iterations=20000,
                               learning_rate=0.067,
                               l2_leaf_reg=17,
                               bootstrap_type='Bernoulli',
                               depth=2,
                               subsample=0.53,
                               eval_metric='AUC',
                               random_seed=seed,
                               # bagging_temperature = 0.2,
                               # od_type='Iter',
                               # metric_period = 75,
                               task_type = "GPU",
                               early_stopping_rounds=1500,
                               )
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),

              cat_features=None,
              use_best_model=True,
              verbose=True)

    # Output .pkl model
    model_name = "CatBoost_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '_MODEL.pkl'
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
