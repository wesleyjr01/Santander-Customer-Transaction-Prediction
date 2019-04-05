import pandas as pd
import pickle
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import datetime
import glob
import numpy as np

def ensemble_models(df):
    """
        This Function is supposed to read all files in local dir with patter '*MODEL.pkl',
        load these models, append them in a vector, then avegare them.
    """

    # os.chdir("/mydir")
    y_pred = np.zeros(test.shape[0])  # Vector to be filled
    for file in glob.glob("*MODEL.pkl"):
        print(file)
        model = pickle.load(open(file, 'rb'))
        if 'LGB' in file: # Load LGB Models
            y_pred += np.array(model.predict(df))
        elif 'XGB' in file: # Load XGBoost Models
            y_pred += np.array(model.predict(xgb.DMatrix(df.values)))
        elif 'CatBoost' in file:
            y_pred += np.array(model.predict_proba(df)[:, 1])
    lenn = len(glob.glob("*MODEL.pkl"))
    print(f'len: {lenn}')
    y_pred /= len(glob.glob("*MODEL.pkl"))
    return y_pred

test = pd.read_pickle("test.pkl")
test_id = test['ID_code'].values
test.drop(['ID_code'], axis=1, inplace=True)
y_pred = ensemble_models(df=test)

submission = pd.DataFrame({"ID_code": test_id})
submission['target'] = y_pred
filename = "ENSEMBLE_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".csv"
filename = filename.replace(':', '_')
submission.to_csv(filename, index=False)
