import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb
import pickle
import datetime

train = pd.read_pickle("train.pkl").drop("ID_code", axis=1)
test = pd.read_pickle("test.pkl")
test_id = test['ID_code'].values
test.drop(['ID_code'], axis=1, inplace=True)


MAX_TREE_DEPTH = 2
LEARNING_RATE = 0.017
TREE_METHOD = 'hist'
ITERATIONS = 100000
SUBSAMPLE = 0.57
REGULARIZATION = 0.23
GAMMA = 0.3
POS_WEIGHT = 1
EARLY_STOP = 1000
params = {'tree_method': TREE_METHOD, 'max_depth': MAX_TREE_DEPTH, 'alpha': REGULARIZATION,
          'gamma': GAMMA, 'subsample': SUBSAMPLE, 'scale_pos_weight': POS_WEIGHT, 'learning_rate': LEARNING_RATE,
          'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True,
          'verbose_eval': False}


result = np.zeros(test.shape[0])  # Vector to be filled
oof = np.zeros(len(train))  # Vector to be filled
for splits in [2, 4, 6]:
  rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=3, random_state=17)
  for counter, (train_index, valid_index) in enumerate(rskf.split(train, train.target), 1):
      print(counter)

      # Train data
      t = train.iloc[train_index]
      trn_data = xgb.DMatrix(t.drop('target', axis=1).values,
                             t.target.values)

      # Validation data
      v = train.iloc[valid_index]
      val_data = xgb.DMatrix(v.drop('target', axis=1).values,
                             v.target.values)

      # Training
      model = xgb.train(params, trn_data, ITERATIONS, evals=[(trn_data, "train"), (val_data, "eval")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=1)

      # Output .pkl model
      model_name = "XGB_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.pkl'
      model_name = model_name.replace(':', '_')
      pickle.dump(model, open(model_name, 'wb'))

      # Feed OOF Vector with Val Prediction
      oof[valid_index] = model.predict(xgb.DMatrix(v.drop('target', axis=1).values))

      # Feed Test Prediction Vector
      result += model.predict(xgb.DMatrix(test.values))

submission = pd.DataFrame({"ID_code": test_id})
submission['target'] = result / counter
filename = "XGB_submission" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv"
filename = filename.replace(':', '_')
submission.to_csv(filename, index=False)
