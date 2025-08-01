from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from decision_trees.base_tree import BaseTree
from decision_trees.base_node import Node
from utils.utils import *


class ID3(ABC, BaseTree):
    def __init__(self, numerical_features=(), impurity_criterion='entropy'):
        super().__init__(numerical_features)

        self.impurity_criterion = impurity_criterion

    @abstractmethod
    def get_default_value(self, y: pd.Series) -> Union[str, float]:
        """Implemented by ID3Classifier or ID3Regressor"""

    @abstractmethod
    def _compute_target_impurity(self, y: pd.Series) -> float:
        """Implemented by ID3Classifier or ID3Regressor"""
        pass

    @abstractmethod
    def _compute_feature_information_gain(self, X: pd.Series, y: pd.Series, sample_size: int, y_impurity: float) -> float:
        """Implemented by ID3Classifier or ID3Regressor"""
        pass

    def _compute_feature_impurity(self, X: pd.Series, y: pd.Series, sample_size: int) -> float:
        """
        Compute the impurity for a given feature.

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

        Returns:
        -------
        entropy : float
            Impurity resulting from splitting on the given feature.
        """

        impurity = 0
        feature_observations = X.unique()
        for observation in feature_observations:
            y_filtered_by_feat_obs = y[X.isin([observation])]
            y_filtered_by_feat_obs_entropy = self._compute_target_impurity(y_filtered_by_feat_obs)
            proba_weight = X.value_counts()[observation] / sample_size
            impurity += proba_weight * y_filtered_by_feat_obs_entropy
        return impurity

    def _handle_continuous_features(self, X: pd.Series, y: pd.Series, sample_size: len, y_impurity: float) -> tuple[float, float]:
        feat_sorted_values = X.sort_values().values
        possible_split_points = (feat_sorted_values[1:] + feat_sorted_values[:-1]) / 2

        best_split_point = np.nan
        best_split_point_information_gain = -np.inf
        for split_point in possible_split_points:
            X_feature_split = (X <= split_point).astype(str)
            information_gain_on_split = self._compute_feature_information_gain(X_feature_split, y, sample_size, y_impurity)
            if information_gain_on_split > best_split_point_information_gain:
                best_split_point_information_gain = information_gain_on_split
                best_split_point = split_point

        return  best_split_point_information_gain, best_split_point

    def _compute_best_feature(self, X: pd.DataFrame, y: pd.Series) -> tuple[str, Union[float, None]]:
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
        best_split_point : Union[float, None]
            Best split point floating number if best_feature is continuous else None
        """

        sample_size = len(y)
        y_impurity = self._compute_target_impurity(y)

        information_gain_split = {}
        for feature in X.columns:
            if feature in self.numerical_features:
                best_split_point_information_gain, best_split_point = self._handle_continuous_features(X[feature], y, sample_size, y_impurity)

                information_gain_split[feature] = {'information_gain': best_split_point_information_gain,
                                                   'best_split_point': best_split_point}

            else:
                information_gain_split[feature] = {'information_gain': self._compute_feature_information_gain(X[feature], y, sample_size, y_impurity),
                                                   'best_split_point': None}

        # Get the key (feature) with the highest value (information gain)
        best_feature = max(information_gain_split, key=lambda x: information_gain_split[x]['information_gain'])
        return best_feature, information_gain_split[best_feature]['best_split_point']

    def _filter_df_by_dict(self, X: pd.DataFrame, y: pd.Series, relations_dict: dict) -> tuple[pd.DataFrame, pd.Series]:
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
            A dictionary where keys are column names in `X` and values are the CustomLambda functions
            with the info if a sample should be filtered or not.

        Returns:
        -------
        X_filtered : pd.DataFrame
            The filtered DataFrame with rows matching the criteria and columns from `relations_dict` removed.
        y_filtered : pd.Series or pd.DataFrame
            The filtered target values corresponding to the retained rows in `X_filtered`.
        """

        condition = X.apply(lambda x: all(filter_rel(x[feat]) for feat, filter_rel in relations_dict.items()), axis=1)
        X_filtered = X[condition].drop(list(set(relations_dict) - set(self.numerical_features)), axis=1)
        y_filtered = y[condition]

        return X_filtered, y_filtered

    @staticmethod
    def _get_feat_observations_visited_nodes(X: pd.DataFrame, root_node_feat: str, root_node_split_point: Union[float, None]) -> tuple:
        """
        Determines the observations and visited nodes for a given feature in a decision tree.

        Parameters:
        ----------
        X : pd.DataFrame
            The input dataset containing feature values.
        root_node_feat : str
            The feature at the root node of the current subtree.
        root_node_split_point : float or None
            The split point value for the feature. If None, the feature is categorical.

        Returns:
        -------
        visited_nodes : list
            A list of features (or nodes) that have been visited. Contains the root feature if the
            feature is categorical, otherwise empty.
        feature_observations : list or tuple
            The unique values of the feature if it is categorical, otherwise (True, False).
        """

        if root_node_split_point is None:
            visited_nodes = [root_node_feat]
            feature_observations = X[root_node_feat].unique()
        else:
            visited_nodes = []
            feature_observations = (True, False)

        return visited_nodes, feature_observations

    @staticmethod
    def _get_conditional_X_y(X: pd.DataFrame, y: pd.Series, root_node_split_point: Union[float, None], root_node_feat: str, observation: str, visited_nodes: list) -> tuple:
        """
        Filters the dataset based on the current observation at a node in the decision tree.

        Parameters:
        ----------
        X : pd.DataFrame
            The input features.
        y : pd.Series or pd.DataFrame
            The target variable(s).
        root_node_split_point : float or None
            The split value for the feature at the current node. If None, the feature is considered
            categorical.
        root_node_feat : str
            The name or index of the feature at the current node.
        observation : any
            The observed value used for conditioning the data.
        visited_nodes : list
            A list of feature names or indices that have already been visited and should be excluded
            from the resulting dataset.

        Returns:
        -------
        conditional_X : pd.DataFrame
            The subset of `X` filtered based on the current node’s condition and with visited
            features dropped.
        conditional_y : pd.Series or pd.DataFrame
            The subset of `y` corresponding to the filtered rows of `X`.
        """

        if root_node_split_point is None:
            condition = X[root_node_feat].isin([observation])
        else:
            condition = (X[root_node_feat] <= root_node_split_point).isin([observation])

        conditional_X = X[condition].drop(visited_nodes, axis=1)
        conditional_y = y[condition]

        return conditional_X, conditional_y

    @staticmethod
    def _update_root_and_leaf_nodes(root_node: 'Node', root_node_feat: str, root_node_split_point: Union[float, None], leaf_node: 'Node', observation: str) -> None:
        """
        Updates the parent and child relationships between a root node and a new leaf node.

        Parameters:
        ----------
        root_node : Node
            The current root node from which the leaf is derived.
        root_node_feat : str
            The feature used for splitting at the root node.
        root_node_split_point : float or None
            The split threshold if the feature is continuous. If None, the feature is treated as
            categorical.
        leaf_node : Node
            The new leaf node to be connected as a child of the root.
        observation : any
            The observed feature value or condition result (True/False) used to determine the path
            from the root to the leaf.

        Returns:
        -------
        None
        """

        if root_node_split_point is None:
            repr_ = observation
            filter_relation = CustomLambda(lambda x: x == observation, observation)
        else:
            repr_ = f'<= {root_node_split_point}' if observation else f'> {root_node_split_point}'
            filter_relation = CustomLambda(lambda x: x <= root_node_split_point if observation else x > root_node_split_point, repr_)

        root_node.children[repr_] = leaf_node
        leaf_node.parent_path[root_node_feat] = filter_relation
        leaf_node.parent_path.update(root_node.parent_path)

    def _fit_stump(self, X: pd.DataFrame, y: pd.Series, root_node: Node, queue: list) -> None:
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

        Returns:
        -------
        None
        """

        root_node_feat = root_node.feature
        root_node_split_point = root_node.split_point
        visited_nodes, feature_observations = self._get_feat_observations_visited_nodes(X, root_node_feat, root_node_split_point)

        # Loop over all the possible observations available for root_node_feat
        for observation in feature_observations:
            conditional_X, conditional_y = self._get_conditional_X_y(X, y, root_node_split_point, root_node_feat, observation, visited_nodes)
            # If target is pure (all values in conditional_y are the same), then create a leaf
            if conditional_y.nunique() == 1:
                value = conditional_y.unique()[0]
                leaf_node = Node(value=value)
                self.leaf_nodes.append(leaf_node)
            # If there are no more features to split (and target is not pure), create a leaf with the most common value
            elif conditional_X.shape[1] == 1:
                best_value = self.get_default_value(conditional_y)
                leaf_node = Node(value=best_value)
                self.leaf_nodes.append(leaf_node)
            # Create a new node using the best feature
            else:
                leaf_node_feat, leaf_node_best_split_point = self._compute_best_feature(conditional_X, conditional_y)
                default_value = self.get_default_value(conditional_y)
                leaf_node = Node(feature=leaf_node_feat, value=default_value, split_point=leaf_node_best_split_point)
                visited_nodes.append(leaf_node_feat)
                queue.append(leaf_node)  # Append the new node (stump's leaf node) to the end of the queue

            # Update the path of the leaf node and save it as the children of root node
            self._update_root_and_leaf_nodes(root_node, root_node_feat, root_node_split_point, leaf_node, observation)

    @staticmethod
    def _set_functional_predict(node: 'Node') -> None:
        """
        Sets the functional prediction logic for a given node.

        This method assigns a function to the node that represents the decision
        rule at that point in the tree. If the node does not have a split point (i.e., it's categorical),
        it assigns the identity function. Otherwise, it creates a binary string function that
        distinguishes between values less than or equal to the split point and those greater.

        Parameters:
        ----------
        node : Node
        The decision tree node for which to set the prediction function.

        Returns:
        -------
        None
        """

        split_point = node.split_point
        if split_point is None:
            predict_relation = CustomLambda(lambda x: x, 'Identity')
        else:
            predict_relation = CustomLambda(lambda x: f'<= {split_point}' if x <= split_point else f'> {split_point}', f'<= {split_point}')

        node.predict_relation = predict_relation

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
        node_feat, best_split_point = self._compute_best_feature(X, y)
        self.tree = Node(feature=node_feat, split_point=best_split_point)

        # We grow the three level-wise
        queue = [self.tree]
        while queue:
            node = queue.pop(0)
            # Set the predict function based on the feature type (Categorical or Continuous)
            self._set_functional_predict(node)
            # Filter the dataframe and target, using the path from root to the node
            X_filtered, y_filtered = self._filter_df_by_dict(X_, y_, node.parent_path)
            # Fit a stump with root_node = node
            self._fit_stump(X_filtered, y_filtered, node, queue)

    def _predict_one_sample(self, x: pd.Series) -> str:
        """
        Predicts the target value for a single sample by traversing the decision tree.

        Starting from the root node, this method follows the decision path defined by the
        feature values in the input sample `x`, descending through the tree until it reaches
        a leaf node. If the input observation is not known, defaults to the most common output value
        in the decision path.
        The prediction is the value stored in that leaf node.

        Parameters:
        ----------
        x : pd.Series
            A single sample with feature names as the index and corresponding values.

        Returns:
        -------
        str :
            The predicted target value stored in the reached leaf node.
        """

        node = self.tree
        node_feat = node.feature
        observation = x[node_feat]
        while True:
            node_ = node.children.get(node.predict_relation(observation))
            if node_:
                node = node_
            else:
                logger.warning(f'{observation} is not a known observation of {node}! Defaulting to {node.value}.')
                return node.value

            if node.feature is None:
                return node.value
            observation = x[node.feature]

    def predict(self, X: pd.DataFrame) -> list:
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
