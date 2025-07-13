class Node:
    def __init__(self, value=None, feature=None, split_point=None, predict_relation=None):
        self.value = value
        self.feature = feature
        self.children = {}
        self.parent_path = {}
        self.split_point = split_point
        self.predict_relation = predict_relation

    def __repr__(self):
        if self.feature:
            return f'Node(feature={self.feature}, value={self.value})'
        else:
            return f'Node(value={self.value})'
