import numpy as np
import pandas as pd

class ID3Classifier:
    def __init__(self, impurity_criterion='entropy'):
        self.tree = None
        self.impurity_criterion = impurity_criterion
        self.leaf_nodes = []

    def _compute_feature_information_gain(self, X: pd.DataFrame, y: pd.Series, feat: str, sample_size: int, y_entropy: float) -> float:
        """
        Compute the information gain for a given feature.

        Parameters:
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series or pd.DataFrame
            The target variable(s).
        feat : str
            Feature for which to compute information gain.
        sample_size : int
            Number of samples to consider from X and y.
        y_entropy : float
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
            impurity =  self._compute_feature_entropy(X, y, feat, sample_size)
        else:
            raise NotImplementedError(f'Impurity Criterion {self.impurity_criterion} not Implemented!')

        information_gain = y_entropy - impurity
        return information_gain

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
            raise NotImplementedError('Impurity Criterion {self.impurity_criterion} not Implemented!')

        return impurity

    def _compute_feature_entropy(self, X: pd.DataFrame, y: pd.Series, feat: str, sample_size: int) -> float:
        """
        Compute the entropy for a given feature.

        Parameters:
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series or pd.DataFrame
            The target variable(s).
        feat : str
            Feature for which to compute information gain.
        sample_size : int
            Number of samples to consider from X and y.

        Returns:
        -------
        entropy : float
            Entropy resulting from splitting on the given feature.
        """

        entropy = 0
        feature_observations = X[feat].unique()
        for observation in feature_observations:
            y_filtered_by_feat_obs = y[X[feat].isin([observation])]
            y_filtered_by_feat_obs_entropy = self._compute_target_impurity(y_filtered_by_feat_obs)
            proba_weight = X[feat].value_counts()[observation] / sample_size
            entropy += proba_weight * y_filtered_by_feat_obs_entropy
        return entropy

    def _compute_best_feature(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Determines the best feature to split on by computing each feature information gain based on the
        impurity_criterion.

        This method evaluates all features in the dataset and calculates their information gain with respect to the target `y`.
        The feature with the lowest impurity (highest information gain) is selected as the best feature for splitting.

        Parameters:
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series or pd.DataFrame
            The target variable(s).

        Returns:
        -------
        best_feature : str
            The name of the feature with the lowest impurity.
        """

        sample_size = len(y)
        y_impurity = self._compute_target_impurity(y)

        information_gain = {}
        for feature in X.columns:
            information_gain[feature] = self._compute_feature_information_gain(X, y, feature, sample_size, y_impurity)

        # Get the key (feature) with the highest value (information gain)
        best_feature = max(information_gain, key=information_gain.get)
        return best_feature

    @staticmethod
    def _filter_df_by_dict(X: pd.DataFrame, y: pd.Series, relations_dict: dict) -> tuple[pd.DataFrame, pd.Series]:
        """
        Filters rows in a DataFrame `X` and its corresponding labels `y` based on a dictionary of column-value pairs.

        This function selects only the rows in `X` where each specified column matches the corresponding value
        in `relations_dict`. After filtering, the columns specified in `relations_dict` are dropped from `X`.
        The corresponding entries in `y` are filtered to match the selected rows.

        Parameters:
        ----------
        X : pd.DataFrame
            The input features DataFrame.
        y : pd.Series or pd.DataFrame
            The target values corresponding to the rows of `X`.
        relations_dict : dict
            A dictionary where keys are column names in `X` and values are the required values
            those columns must match to retain a row.

        Returns:
        -------
        X_filtered : pd.DataFrame
            The filtered DataFrame with rows matching the criteria and columns from `relations_dict` removed.
        y_filtered : pd.Series or pd.DataFrame
            The filtered target values corresponding to the retained rows in `X_filtered`.
        """

        condition = (X[list(relations_dict)] == pd.Series(relations_dict)).all(axis=1)
        X_filtered = X[condition].drop(relations_dict, axis=1)
        y_filtered = y[condition]

        return X_filtered, y_filtered

    def _fit_stump(self, X: pd.DataFrame, y: pd.Series, root_node: 'Node', queue: list) -> None:
        """
        Fits a one-level decision tree (stump) starting from the given root node.

        This method expands the root node by iterating over the unique values of the root feature,
        creating child nodes for each unique value. Depending on the class distribution of the
        corresponding subset of data, it either creates a leaf node (if pure or only one feature
        remains) or creates an internal node to be further split later.

        Parameters:
        ----------
        X : pd.DataFrame
            The input feature data.
        y : pd.Series
            The target variable corresponding to the rows in X.
        root_node : Node
            The current node from which to expand the stump.
        queue : list
            A queue to collect internal nodes that need further splitting.

        Side Effects:
            - Updates the `children` attribute of `root_node` with new leaf or internal nodes.
            - Adds newly created leaf nodes to `self.leaf_nodes`.
            - Appends internal nodes to the provided `queue` for further processing.
            - Updates the `parent_path` attribute of each created node to track its decision path.

        Note:
            Assumes the `Node` object has attributes `feature`, `value`, `children`, and `parent_path`.
        """

        root_node_feat = root_node.feature
        visited_nodes = []
        feature_observations = X[root_node_feat].unique()
        # Loop over all the possible observations available for root_node_feat
        for observation in feature_observations:
            condition = X[root_node_feat].isin([observation])
            conditional_X = X[condition].drop(visited_nodes, axis=1)
            conditional_y = y[condition]
            # If target is pure (all values in conditional_y are the same), then create a leaf
            if conditional_y.nunique() == 1:
                value = conditional_y.unique()[0]
                leaf_node = Node(value=value)
                self.leaf_nodes.append(leaf_node)
            # If there are no more features to split (and target is not pure), create a leaf with the most common value
            elif conditional_X.shape[1] == 1:
                best_value = conditional_y.mode()[0]
                leaf_node = Node(value=best_value)
                self.leaf_nodes.append(leaf_node)
            # Create a new node using the best feature
            else:
                leaf_node_feat = self._compute_best_feature(conditional_X, conditional_y)
                leaf_node = Node(feature=leaf_node_feat)
                visited_nodes.append(leaf_node_feat)
                queue.append(leaf_node)  # Append the new node (stump's leaf node) to the end of the queue

            # Update the path of the leaf node and save it as the children of root node
            leaf_node.parent_path[root_node_feat] = observation
            leaf_node.parent_path.update(root_node.parent_path)
            root_node.children[observation] = leaf_node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits a decision tree model using the ID3 algorithm.

        The method creates the root node based on the best splitting feature, then grows the tree
        level by level (resulting in a balanced structure). It uses a queue to track nodes that
        need to be expanded. At each node, it filters the data based on the path from the root to
        that node and fits a shallow decision tree (stump) rooted at the current node.

        Parameters:
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series or pd.DataFrame
            The target variable(s).

        Returns:
        -------
        None
        """

        X_ = X.copy()
        y_ = y.copy()

        # Create root node
        node_feat = self._compute_best_feature(X, y)
        self.tree = Node(feature=node_feat)

        # We grow the three level-wise, resulting in a balanced tree
        # We do this by creating a queue (First In First Out)
        queue = [self.tree]
        while queue:
            node = queue.pop(0)
            # Filter the dataframe and target, using the path from root to the node
            X_filtered, y_filtered = self._filter_df_by_dict(X_, y_, node.parent_path)
            # Fit a stump with root_node = node
            self._fit_stump(X_filtered, y_filtered, node, queue)

    def _predict_one_sample(self, x: pd.Series) -> 'Node':
        """
        Predicts the target value for a single sample by traversing the decision tree.

        Starting from the root node, this method follows the decision path defined by the
        feature values in the input sample `x`, descending through the tree until it reaches
        a leaf node. The prediction is the value stored in that leaf node.

        Parameters:
        ----------
        x : pd.Series
            A single sample with feature names as the index and corresponding values.

        Returns:
        -------
        str :
            The predicted target value stored in the reached leaf node.

        Assumes:
            - `self.tree` is the root node of a decision tree with properly populated `children`.
            - Each node has attributes `feature`, `children`, and `value`.
            - The traversal ends when a node with a non-None `value` is encountered (i.e., a leaf).
        """

        node = self.tree
        node_feat = node.feature
        node_feat_name = x[node_feat]
        while True:
            node = node.children[node_feat_name]
            if node.value:
                return node.value
            node_feat_name = x[node.feature]

    def predict(self, X):
        """
        Predicts target values for a given dataset using the fitted decision tree.

        This method applies the `_predict_one_sample` function to each row in the input
        DataFrame `X`, traversing the tree to return the predicted value for each sample.

        Parameters:
        ----------
        X : pd.DataFrame
            A DataFrame where each row represents a sample with feature values.

        Returns:
        -------
        predict : list
            A list of predicted target values, one for each sample in `X`.

        Raises:
            AssertionError: If the tree has not been fitted (`self.tree` is None).
        """

        assert self.tree is not None, 'Tree not fitted!'
        predict = X.apply(self._predict_one_sample, axis=1).to_list()
        return predict

    def transform(self):
        pass


class Node:
    def __init__(self, value=None, feature=None):
        self.value = value
        self.feature = feature
        self.children = {}
        self.parent_path = {}

    def __repr__(self):
        if self.value:
            return f'Node(value={self.value})'
        else:
            return f'Node(feature={self.feature})'


if __name__ == '__main__':
    def create_dataset():
        data = {
            "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
            "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
            "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
            "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
            "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
        }

        data_test = {
            "Outlook": ["Sunny", "Rain"],
            "Temperature": ["Hot", "Mild"],
            "Humidity": ["High", "High"], # "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
            "Wind": ["Weak", "Strong"], #, "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
            "Play": ["No", "Yes"]
        }

        return pd.DataFrame(data), pd.DataFrame(data_test)

    df_train, df_test = create_dataset()

    X_train, y_train = df_train.drop(['Play'], axis=1), df_train['Play']
    X_test, y_test = df_test.drop(['Play'], axis=1), df_test['Play']

    tree = ID3Classifier()
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
