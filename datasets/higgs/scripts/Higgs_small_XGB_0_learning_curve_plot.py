import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import TrainingCallback
import matplotlib.pyplot as plt

X_num_train = np.load('../input/higgs_small_roc/X_num_train.npy')
X_num_val = np.load('../input/higgs_small_roc/X_num_val.npy')
X_num_test = np.load('../input/higgs_small_roc/X_num_test.npy')

y_train = np.load('../input/higgs_small_roc/y_train.npy')
y_val = np.load('../input/higgs_small_roc/y_val.npy')
y_test = np.load('../input/higgs_small_roc/y_test.npy')

# Enable interactive plotting
plt.ion()

class LivePlotCallback(TrainingCallback):
    def __init__(self):
        self.train_metrics = []
        self.eval_metrics = []
        self.fig, self.ax = plt.subplots()

    def after_iteration(self, model, epoch, evals_log):
        """
        This method is called after each training iteration.
        
        Parameters:
          model: The Booster model.
          epoch: The current boosting round.
          evals_log: A dict containing evaluation results. For example:
                     {'train': {'error': [0.1, 0.05, ...]},
                      'eval': {'error': [0.15, 0.07, ...]}}
        """
        # Extract the latest evaluation results; adjust metric name if needed.
        train_error = evals_log.get('train', {}).get('error', [None])[-1]
        eval_error = evals_log.get('eval', {}).get('error', [None])[-1]
        
        self.train_metrics.append(train_error)
        self.eval_metrics.append(eval_error)
        
        # Clear the axes and re-plot
        self.ax.cla()
        self.ax.plot(self.train_metrics, label='Train')
        self.ax.plot(self.eval_metrics, label='Eval')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Error')
        self.ax.legend()
        
        plt.draw()
        plt.pause(0.001)  # Short pause to update the figure
        
        # Returning False means "continue training"
        return False
    
# Create an instance of the callback
live_plot = LivePlotCallback()

# Prepare your data
dtrain = xgb.DMatrix(X_num_train, label=y_train)
deval  = xgb.DMatrix(X_num_val, label=y_val)

# Define your parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
    # add other parameters as needed
}


# Start training with evaluation sets and the custom callback.
# Here, we pass two evaluation sets: one for training and one for evaluation.
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (deval, 'eval')],
    callbacks=[live_plot]
)