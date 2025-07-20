import numpy as np
import pandas as pd

from decision_trees.id3.id3 import ID3


class ID3Classifier(ID3):
    def __init__(self, numerical_features=(), impurity_criterion='entropy'):
        super().__init__(numerical_features)
        self.impurity_criterion = impurity_criterion

    def _get_task(self) -> str:
        return 'classification'

    def _compute_target_impurity(self, y: pd.Series) -> float:
        """
        Compute the impurity of the target variable y.

        Parameters:
        ----------
        y : pd.Series or pd.DataFrame
            The target variable(s).

        Returns:
        -------
        impurity : float
            Impurity resulting from splitting on the given feature.

        Raises:
        ------
        NotImplementedError: If the impurity criterion is not 'entropy'.
        """

        if self.impurity_criterion == 'entropy':
            probas = (y.value_counts() / len(y)).values
            log_probas = np.log2(probas)
            impurity = -sum(probas * log_probas)
        else:
            raise NotImplementedError(f'Impurity Criterion {self.impurity_criterion} not Implemented!')

        return impurity

    def _compute_feature_information_gain(self, X: pd.Series, y: pd.Series, sample_size: int, y_impurity: float) -> float:
        """
        Compute the information gain for a given feature.

        Parameters:
        ----------
        X : pd.Series
            Observations of a given feature.
        y : pd.Series or pd.DataFrame
            The target variable(s).
        feat : str
            Feature for which to compute information gain.
        sample_size : int
            Number of samples to consider from X and y.
        y_impurity : float
            Entropy of the current node (before the split).

        Returns:
        -------
        information_gain : float
            Entropy resulting from splitting on the given feature.

        Raises:
        ------
        NotImplementedError: If the impurity criterion is not 'entropy'.
        """

        if self.impurity_criterion == 'entropy':
            impurity =  self._compute_feature_impurity(X, y, sample_size)
        else:
            raise NotImplementedError(f'Impurity Criterion {self.impurity_criterion} not Implemented!')

        information_gain = y_impurity - impurity
        return information_gain


if __name__ == '__main__':
    def create_dataset():
        data = {
             "Outlook"     : ["Sunny", "Rain", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
             "Temperature" : [30, 35, 3, 23, 15, 14, 14, 21, 10, 22, 23, 20, 28, 19],
             "Humidity"    : ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
             "Wind"        : ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
             "Play"        : ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
        }

        data_test = {
             "Outlook"     : ["Sunny", "Rain"],
             "Temperature" : [32, 24],
             "Humidity"    : ["High", "High"],
             "Wind"        : ["Weak", "Strong"],
             "Play"        : ["No", "No"]
        }

        return pd.DataFrame(data), pd.DataFrame(data_test)

    df_train, df_test = create_dataset()

    X_train, y_train = df_train.drop(['Play'], axis=1), df_train['Play']
    X_test, y_test = df_test.drop(['Play'], axis=1), df_test['Play'].to_list()
    print('True y:  ', y_test)

    tree = ID3Classifier(numerical_features=('Temperature',))
    tree.fit(X_train, y_train)
    print('Predict: ', tree.predict(X_test))
    tree.visualize_tree()
