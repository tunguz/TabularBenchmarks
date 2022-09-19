from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask import dataframe as dd
from dask.delayed import delayed
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import optuna
import gc
import logging
print(xgb.__version__)

num_round = 1000

def objective(client, train, target, train_folds, val_folds, train_ys, val_ys, trial):
        
    params = {
        'objective':'multi:softmax', 
        'num_class':10,
        'tree_method': trial.suggest_categorical('tree_method',['gpu_hist']),  # 'gpu_hist','hist'
        'lambda': trial.suggest_loguniform('lambda',1e-4,10.0),
        'alpha': trial.suggest_loguniform('alpha',1e-4,10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3,1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001,0.1),
        #'n_estimators': trial.suggest_categorical('n_estimators', [1000]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 5,7,9,11,13,15,17,20]),
        #'random_state': trial.suggest_categorical('random_state', [24,48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1,300),
        'eval_metric': trial.suggest_categorical('eval_metric',['logloss']),
        'random_state': 137,

    }
    
    #start_time = time()
    
    train_oof = np.zeros((train.shape[0],))
    kf = StratifiedKFold(5, shuffle=True, random_state=1974)

    for i, (train_index, val_index) in enumerate(kf.split(train,target)):
        dtrain = xgb.dask.DaskDMatrix(client, train_folds[i], train_ys[i])
        dval = xgb.dask.DaskDMatrix(client, val_folds[i], val_ys[i])
        
        output = xgb.dask.train(client, params, dtrain, num_round)
        booster = output['booster']  # booster is the trained model
        booster.set_param({'predictor': 'gpu_predictor'})
        predictions = xgb.dask.predict(client, booster, dval)
        predictions = predictions.compute()
        train_oof[val_index] = predictions
        del dtrain, dval, output
        gc.collect()
        gc.collect()


    acc = accuracy_score(target, train_oof)
    
    return acc

def main():
    
    
    with LocalCUDACluster(n_workers=4) as cluster:
        client = Client(cluster)
        
        train = pd.read_csv('../input/train.csv.gz')
        target = train.label.values
        train_oof = np.zeros((train.shape[0],))
        
        train_folds = []
        val_folds = []
        train_ys = []
        val_ys = []

        for i in range(5):
            print(f'Loading fold {i}')
            train_fold_d = delayed(pd.read_csv)(f'../input/xgtrain_fold_{i}.csv.gz')
            train_fold = dd.from_delayed(train_fold_d)

            val_fold_d = delayed(pd.read_csv)(f'../input/xgval_fold_{i}.csv.gz')
            val_fold = dd.from_delayed(val_fold_d)


            train_y = train_fold['target']
            train_fold = train_fold[train_fold.columns.difference(['target'])]

            val_y = val_fold['target']
            val_fold = val_fold[val_fold.columns.difference(['target'])]

            train_folds.append(train_fold)
            val_folds.append(val_fold)

            train_ys.append(train_y)
            val_ys.append(val_y)





        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler("optuna_xgb_output_4.log", mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

        study = optuna.load_study(storage="sqlite:///xgb_optuna_mnist_0.db",
                                    study_name="five_fold_optuna_xgb_0")
        logger.info("Start optimization.")
        #study.optimize(objective, n_trials=2)
        study.optimize(lambda trial: objective(client, train, target, train_folds, val_folds, train_ys, val_ys, trial), n_trials=1000)

    #df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    #df.to_csv('optuna_xgb_output_2.csv', index=False)
    

    
    
if __name__ == "__main__":
    main()
