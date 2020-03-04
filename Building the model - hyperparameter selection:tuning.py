import pandas as pd
from surprise import Reader,Dataset, accuracy, dump, SVD
import surprise
import pickle
from surprise.model_selection import cross_validate, GridSearchCV, KFold

data = pd.read_pickle("data.pickle")

algo = surprise.SVD(n_factors=15, n_epochs=10, lr_all=0.03, reg_all=0.04, verbose=True)
kf = KFold(n_splits=3)

for trainset, testset in kf.split(data):
    pickle_out = open("testset.pickle", "wb")
    pickle.dump(testset, pickle_out)
    pickle_out.close()
    algo.fit(trainset)
    dump.dump('saved svd modelV12', predictions=False, algo=algo, verbose=0)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
