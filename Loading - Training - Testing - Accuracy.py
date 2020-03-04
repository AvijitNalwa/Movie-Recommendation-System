import pandas as pd
from surprise import Reader,Dataset, accuracy, dump, SVD
import surprise
import pickle
from surprise.model_selection import cross_validate, GridSearchCV, KFold, train_test_split

data = pd.read_pickle("data.pickle")
test = pd.read_pickle("test.pickle")
kf = KFold()

trainset, testset = train_test_split(data, test_size=.80)


algo = dump.load('saved svd modelV12')
model = algo[1].fit(data.build_full_trainset())
predictions = algo[1].test(testset)
dump.dump('Complete SVD v1.12', predictions=False, algo=algo, verbose=0)
accuracy.rmse(predictions, verbose=True)
accuracy.mae(predictions, verbose=True)
accuracy.fcp(predictions, verbose=True)
accuracy.mse(predictions, verbose=True)
