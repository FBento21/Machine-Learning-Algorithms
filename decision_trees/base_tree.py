from io import BytesIO
from typing import Union

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from graphviz import Digraph
from PIL import Image


class BaseTree:
    def __init__(self, numerical_features: tuple, task: Union[None, str] = None):
        assert task in ('classification', 'regression')

        self.tree = None
        self.leaf_nodes = []
        self.numerical_features = numerical_features
        self.task = task

    def get_default_value(self, y: pd.Series) -> Union[str, float]:
        """
        Get default value of an intermediate node

        Parameters:
        ----------
        y : pd.Series
            Conditioned target value

        Returns:
        -------
        Mode of y if classification task, else its mean
        """
        return y.mode()[0] if self.task == 'classification' else y.mean()

    @staticmethod
    def _render_tree(decision_tree: Digraph) -> None:
        """
        Renders a decision tree diagram from a Graphviz Digraph object.

        The method generates a PNG image of the tree and displays it inline using matplotlib.

        Parameters:
        ----------
        decision_tree : Digraph
            A Graphviz Digraph object representing the decision tree structure.

        Returns:
        -------
        None
        """

        # Render to bytes in memory
        img_bytes = decision_tree.pipe(format='png')
        image = Image.open(BytesIO(img_bytes))

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def visualize_tree(self) -> None:
        """
        Visualizes the tree structure using Graphviz and matplotlib.

        This method creates a visual representation of the fitted tree, where each node is
        labeled with its feature or predicted value. It uses a breadth-first search traversal
        to render the nodes and edges level-by-level.

        Raises:
        ------
        AssertionError:
            If the tree has not been fitted (i.e., self.tree is None).

        Returns:
        -------
        None
        """

        assert self.tree is not None, "Cannot graph tree, as it is not fitted!"

        decision_tree = Digraph('Tree', comment='Tree', engine='dot')
        decision_tree.attr(rankdir='TB')
        decision_tree.attr(splines='false')
        decision_tree.attr(nodesep='1')

        base_node_index = 0
        node_index = 0

        base_node = self.tree
        decision_tree.node(str(node_index), base_node.feature, width='0.5', height='0.5', fontsize='10')
        queue = [base_node]

        while queue:
            node = queue.pop(0)
            for edge_name, children_node in node.children.items():
                node_index += 1
                node_label = children_node.feature if children_node.feature else children_node.value
                decision_tree.node(str(node_index), str(node_label), width='0.5', height='0.5', fontsize='10')
                decision_tree.edge(str(base_node_index), str(node_index), label=edge_name, fontsize='10')
                queue.append(children_node)

            base_node_index += 1

        self._render_tree(decision_tree)
