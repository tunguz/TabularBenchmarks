import pandas as pd
import xgboost as xgb
from time import time

t1 = time()
test = pd.read_csv(f'../input/X_test_df_1000.csv.zip')
t2 = time()
print(test.shape)
print(f'Loading the test file took {t2-t1} seconds')

dtest = xgb.DMatrix(test.values, enable_categorical=True)

t1 = time()
bst = xgb.Booster()
model_path = 'model.json'
bst.load_model(model_path)
t2 = time()

print(f'Loading the XGBoost model file took {t2-t1} seconds')

t1 = time()
test_predictions = bst.predict(dtest)
t2 = time()

print(f'XGBoost inference took {t2-t1} seconds')