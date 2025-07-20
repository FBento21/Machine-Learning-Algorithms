import pandas as pd

from decision_trees.id3.id3 import ID3

class ID3Regressor(ID3):
    def __init__(self, numerical_features=(), impurity_criterion='std'):
        super().__init__(numerical_features)
        self.impurity_criterion = impurity_criterion

    def _get_task(self) -> str:
        return 'regression'

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

        if self.impurity_criterion == 'std':
            impurity = y.std(ddof=0)
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

        if self.impurity_criterion == 'std':
            impurity =  self._compute_feature_impurity(X, y, sample_size)
        else:
            raise NotImplementedError(f'Impurity Criterion {self.impurity_criterion} not Implemented!')

        information_gain = y_impurity - impurity
        return information_gain


if __name__ == '__main__':
    def create_dataset():
        data = {
            "Outlook"      : [ "Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
            "Temperature"  : [ 32, 31, 40, 25, 16, 15, 11, 22, 16, 20, 21, 21, 28, 23],
            "Humidity"     : [ "High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
            "Wind"         : [ "Weak", "Cold", "Weak", "Weak", "Weak", "Cold", "Cold", "Weak", "Weak", "Weak", "Cold", "Cold", "Weak", "Cold"],
            "Hours Played" : [ 25, 30, 46, 45, 52, 23, 43, 35, 38, 46, 48, 52, 44, 30]
        }

        data_test = {
            "Outlook"      : ["Sunny", "Rain"],
            "Temperature"  : [31, 24],
            "Humidity"     : ["High", "High"],
            "Wind"         : ["Weak", "Strong"],
            "Hours Played" : [25, 40]
        }

        return pd.DataFrame(data), pd.DataFrame(data_test)

    df_train, df_test = create_dataset()

    X_train, y_train = df_train.drop(['Hours Played'], axis=1), df_train['Hours Played']
    X_test, y_test = df_test.drop(['Hours Played'], axis=1), df_test['Hours Played'].to_list()
    print('True y:  ', y_test)

    tree = ID3Regressor(numerical_features=("Temperature",))
    tree.fit(X_train, y_train)
    print('Predict: ', tree.predict(X_test))
    tree.visualize_tree()

