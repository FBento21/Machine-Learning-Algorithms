import numpy as np
import pandas as pd

def create_dataset():
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny",
                    "Overcast", "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot",
                  "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal",
                     "High", "Normal", "High"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong",
                 "Strong", "Weak", "Strong"],
        "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }

    return pd.DataFrame(data)


class DecisionTreeClassifier:
    def __init__(self):
        pass

    @staticmethod
    def _compute_target_entropy(y):
        probas = (y.value_counts() / len(y)).values
        log_probas = np.log2(probas)
        return -sum(probas * log_probas)

    def _compute_information_gain(self, X, y, feat, size, entropy):
        information_gain = 0
        feature_observations = X[feat].unique()
        for observation in feature_observations:
            y_filtered_by_feat_obs = y[df[feat].isin([observation])]
            y_filtered_by_feat_obs_entropy = self._compute_target_entropy(y_filtered_by_feat_obs)
            proba_weight = X[feat].value_counts()[observation] / size
            information_gain -= proba_weight * y_filtered_by_feat_obs_entropy
        information_gain += entropy
        return information_gain

    def _compute_best_feature(self, X, y):
        size = len(y)
        y_entropy = self._compute_target_entropy(y)

        information_gain_dict = {}
        for feat in X.columns:
            information_gain_dict[feat] = self._compute_information_gain(X, y, feat, size, y_entropy)

        best_feature = max(information_gain_dict, key=information_gain_dict.get)
        return best_feature

    @staticmethod
    def _filter_df_by_dict(X, y, relations_dict):
        condition = (X[list(relations_dict)] == pd.Series(relations_dict)).all(axis=1)
        X_filtered = X[condition].drop(relations_dict, axis=1)
        y_filtered = y[condition]

        return X_filtered, y_filtered

    def _fit_one_tree_level(self, X, y, root_node, queue):
        root_node_feat = root_node.feature
        visited_nodes = []
        for feat in X[root_node_feat].unique():
            condition = X[root_node_feat].isin([feat])
            conditional_X = X[condition].drop(visited_nodes, axis=1)
            conditional_y = y[condition]
            if conditional_y.nunique() == 1:
                value = conditional_y.unique()[0]
                leaf_node = Node(value=value)
            elif conditional_X.shape[1] == 0:
                pass
            else:
                leaf_node_feat = self._compute_best_feature(conditional_X, conditional_y)
                leaf_node = Node(feature=leaf_node_feat)
                queue.append(leaf_node)

            leaf_node.parent_relation[root_node_feat] = feat
            leaf_node.parent_relation.update(root_node.parent_relation)
            root_node.path_to_children[feat] = leaf_node

    def fit(self, X, y):
        X_ = X.copy()
        y_ = y.copy()
        node_feat = self._compute_best_feature(X, y)
        root_node = Node(feature=node_feat)
        queue = [root_node]
        while queue:
            node = queue.pop(0)
            X_filtered, y_filtered = self._filter_df_by_dict(X_, y_, node.parent_relation)
            self._fit_one_tree_level(X_filtered, y_filtered, node, queue)

    def transform(self):
        pass

class Node:
    def __init__(self, value=None, feature=None):
        self.value = value
        self.feature = feature
        self.path_to_children = {}
        self.parent_relation = {}

if __name__ == '__main__':
    df = create_dataset()

    tree = DecisionTreeClassifier()
    tree.fit(df.drop(['Play'], axis=1), df['Play'])

    print('hi')