from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
# from sklearn.model_selection import GridSearchCV


class SVM:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        data = pd.read_csv(self.dataset_path)

        # X-> Contains the features
        self.X = data.iloc[:, 0:-1]
        # y-> Contains all the targets
        self.y = data.iloc[:, -1]

    def solve(self):
        """
        Build an SVM model and fit on the training data
        The data has already been loaded in from the dataset_path

        Refrain to using SVC only (with any kernel of your choice)

        You are free to use any any pre-processing you wish to use
        Note: Use sklearn Pipeline to add the pre-processing as a step in the model pipeline
        Refrain to using sklearn Pipeline only not any other custom Pipeline if you are adding preprocessing

        Returns:
            Return the model itself or the pipeline(if using preprocessing)
        """
        # parameters = {
        #     "classifier__kernel": ["linear", "rbf", "poly", "sigmoid"],
        #     "classifier__C": np.arange(0.1, 2.1, 0.1),
        #     "classifier__gamma": ["scale", "auto"],
        #     "classifier__shrinking": [True, False],
        #     "classifier__tol": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        #     "classifier__class_weight": [None, "balanced"],
        #     "classifier__break_ties": [True, False],
        #     "classifier__degree": [1, 2, 3, 4, 5],
        # }
        # classifier = SVC(max_iter=-1, random_state=0, kernel="rbf")
        # pipe = Pipeline(
        #     [('std', StandardScaler()), ('norm', Normalizer()),
        #      ("classifier", classifier)])
        # search = GridSearchCV(pipe, parameters,
        #                       n_jobs=-1, verbose=3, refit=True)
        # search.fit(self.X, self.y)
        # return search.best_estimator_
        # print(search.best_score_, search.best_params_)
        pipe = Pipeline(
            [('std', StandardScaler()), ('norm', Normalizer()),
             ("classifier", SVC())])
        return pipe.fit(self.X, self.y)
# C=2, tol=0.1
