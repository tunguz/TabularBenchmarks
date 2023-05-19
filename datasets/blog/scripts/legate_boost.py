import cunumeric as cn
import numpy as np
import legateboost as lbst

X_np = np.load('../input/X.npy')
Y_np = np.load('../input/Y.npy')

X = cn.array(X_np)
Y = cn.array(Y_np)

print(X.shape)
print(Y.shape)

model = lbst.LBRegressor(verbose=1, random_state=0, max_depth=2).fit(X,Y)
