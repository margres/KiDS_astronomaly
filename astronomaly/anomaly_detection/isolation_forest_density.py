from astronomaly.base.base_pipeline import PipelineStage
from isotree import IsolationForest
import pandas as pd
import pickle
from os import path


class IforestAlgorithm(PipelineStage):
    def __init__(self, n_estimators=200, **kwargs):
        """
        Runs sklearn's isolation forest anomaly detection algorithm and returns
        the anomaly score for each instance.

        Parameters
        ----------
        n_estimators : int, optional
            Hyperparameter to pass to IsolationForest.

        """
        super().__init__(n_estimators=n_estimators, **kwargs)

        self.n_estimators = n_estimators

        self.iforest_obj = None

    def save_iforest_obj(self):
        """
        Stores the iforest object to the output directory to allow quick 
        rerunning on new data.
        """
        if self.iforest_obj is not None:
            f = open(path.join(self.output_dir, 'isotree_object.pickle'), 'wb')
            pickle.dump(self.iforest_obj, f)

    def _execute_function(self, features):
        """
        Does the work in actually running isolation forest.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The input features to run iforest on. Assumes the index is the id 
            of each object and all columns are to
            be used as features.

        Returns
        -------
        pd.DataFrame
            Contains the same original index of the features input and the 
            anomaly scores. More negative is more anomalous.

        """
        iforest = IsolationForest(n_estimators=self.n_estimators,       
                                  bootstrap=True)
        iforest.fit(features)

        scores = iforest.decision_function(features)

        if self.save_output:
            self.save_iforest_obj()

        return pd.DataFrame(data=scores, index=features.index, 
                            columns=['score'])
