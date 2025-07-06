from io import BytesIO

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from graphviz import Digraph
from PIL import Image


class BaseTree:
    def __init__(self):
        self.tree = None
        self.leaf_nodes = []

    @staticmethod
    def _render_tree(decision_tree):
        # Render to bytes in memory
        img_bytes = decision_tree.pipe(format='png')
        image = Image.open(BytesIO(img_bytes))

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def visualize_tree(self):
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
                decision_tree.node(str(node_index), node_label, width='0.5', height='0.5', fontsize='10')
                decision_tree.edge(str(base_node_index), str(node_index), label=edge_name, fontsize='10')
                queue.append(children_node)

            base_node_index += 1

        self._render_tree(decision_tree)
