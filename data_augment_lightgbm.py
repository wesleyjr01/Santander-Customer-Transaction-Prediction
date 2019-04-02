import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
import datetime
import pickle

train = pd.read_pickle("train.pkl").drop("ID_code", axis=1)
test = pd.read_pickle("test.pkl")
test_id = test['ID_code'].values
test.drop(['ID_code'], axis=1, inplace=True)

# Inspiration from
# https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment


def augment(train, num_n=1, num_p=2):
    newtrain = [train]

    n = train[train.target == 0]
    for i in range(num_n):
        newtrain.append(n.apply(lambda x: x.values.take(np.random.permutation(len(n)))))

    for i in range(num_p):
        p = train[train.target > 0]
        newtrain.append(p.apply(lambda x: x.values.take(np.random.permutation(len(p)))))
    return pd.concat(newtrain)
# df=oversample(train,2,1)


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.37, 'boost_from_average': 'false',
    'boost': 'gbdt', 'feature_fraction': 0.07, 'learning_rate': 0.01,
    'max_depth': -1, 'metric': 'auc', 'min_data_in_leaf': 83, 'min_sum_hessian_in_leaf': 11.0,
    'num_leaves': 11, 'num_threads': 8, 'tree_learner': 'serial', 'objective': 'binary',
    'reg_alpha': 0.1302650970728192, 'reg_lambda': 0.3703427518866501, 'verbosity': 1
}


result = np.zeros(test.shape[0])
oof = np.zeros(len(train))
for splits in [3,5,7]:
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=3, random_state=43)
    for counter, (train_index, valid_index) in enumerate(rskf.split(train, train.target), 1):
        print (counter)

        # Train data
        t = train.iloc[train_index]
        t = augment(t)
        trn_data = lgb.Dataset(t.drop("target", axis=1), label=t.target)

        # Validation data
        v = train.iloc[valid_index]
        val_data = lgb.Dataset(v.drop("target", axis=1), label=v.target)

        # Training
        model = lgb.train(param, trn_data, 100000, valid_sets=[trn_data, val_data],
                        verbose_eval=5000, early_stopping_rounds=2000)
        model_name = 'LGB_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '_MODEL.pkl'
        model_name = model_name.replace(':', '_')
        pickle.dump(model, open(model_name, 'wb')) # wb == write binary
        oof[valid_index] = model.predict(train.iloc[valid_index], num_iteration=model.best_iteration)
        result += model.predict(test)

submission = pd.DataFrame({"ID_code": test_id})
submission['target'] = result / counter
filename = "LGB_submission" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv"
filename = filename.replace(':', '_')
submission.to_csv(filename, index=False)
