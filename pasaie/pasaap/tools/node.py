import os
import json

from .plot import plot_tree


class LogicNode:

    def __init__(self,
                 is_root,
                 logic_type,
                 sentence=None):
        """
            Construct a logic node.
        :param is_root: boolean, if is_root is True, then this node has some sub-nodes. Otherwise, this node only has its corresponding sentence.
        :param logic_type: str, can only be 'AND', 'OR', None
        :param sentence: str, the corresponding sentence of node
        """
        if is_root and logic_type is None:
            raise ValueError('The logic_type can not be None when it is a root-node.')
        if logic_type:
            logic_type = logic_type.upper()
            if logic_type not in ['AND', 'OR']:
                raise ValueError("The logic_type of a sentence-node can only be `AND` or `OR`: {}.".format(logic_type))

        self.logic_type = logic_type
        self.is_root = is_root
        self.sent = sentence
        self.children = {}
        self.parent = None
        self.depth = 0

    def add_node(self, node, node_key=None):
        if node_key is None:
            node_key = 'item{}'.format(len(self.children) + 1)
        self.children[node_key] = node
        node.parent = self
        # update depth
        node.update_depth()

    def update_depth(self):
        if self.parent:
            self.depth = self.parent.depth + 1
        for child_name, child_node in self.children.items():
            child_node.update_depth()

    def convert_to_root_node(self, logic_type, sentence=None):
        """
            Transform a sentence-node to a root-node.
        :param logic_type: str, can only be 'AND', 'OR'.
        :param sentence: str or None.
        :return:
        """
        if logic_type is None:
            raise ValueError('logic_type can not be None.')
        if logic_type:
            logic_type = logic_type.upper()
            if logic_type not in ['AND', 'OR']:
                raise ValueError("The logic_type of a sentence-node can only be `AND` or `OR`: {}.".format(logic_type))
        self.logic_type = logic_type
        self.is_root = True
        self.sent = sentence
        self.children = {}

    def convert_to_leaf_node(self):
        self.logic_type = None
        self.is_root = False
        self.children = {}

    def get_sentence(self):
        if self.sent is None:
            return ''
        elif self.sent and (self.sent.upper().startswith('AND') or self.sent.upper().startswith('OR')):
            return self.sent.split('--', maxsplit=1)[1]
        else:
            return self.sent

    def get_logic_type(self):
        return self.logic_type

    def convert_to_nested_dict(self):
        """
            Transform the logic tree of a article to a nested dictionary.
        :return:
        """
        if not self.is_root:
            raise Exception("Nested dictionary should be converted from a root-node, not a sentence-node.")
        if self.sent:
            key = self.logic_type + '--' + self.sent
        else:
            key = self.logic_type
        nested_dict = {key: {}}
        sub_dict = nested_dict[key]
        for str_idx, node in self.children.items():
            if node.is_root:
                child_dict = node.convert_to_nested_dict()
                sub_dict[str_idx] = child_dict
            else:
                sub_dict[str_idx] = node.sent
        return nested_dict

    def clean_children(self):
        self.children = {}


class LogicTree:

    def __init__(self, root, json_path):
        """
            Construct a logic Tree.
        :param root: LogicNode or None.
        :param json_path: str or None.
        """
        if root:
            self.root = root
        elif json_path:
            self.root = self.construct_tree_from_json(json_path)
        else:
            raise ValueError("Variable `root` and `json_path` cant not both be None!")

        if json_path:
            self.name = json_path.rsplit('/', maxsplit=1)[-1].rsplit('\\', maxsplit=1)[-1].split('.')[0]
        else:
            self.name = 'tree'

    def save_as_json(self, output_path):
        """
            Save as json file.
        :param output_path: str, path of json file
        :return:
        """
        output_dir = output_path.rsplit('/', maxsplit=1)[0].rsplit('\\', maxsplit=1)[0]
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf8') as fout:
            nested_dict = self.root.convert_to_nested_dict()
            json.dump(nested_dict, fout, indent=1, ensure_ascii=False)

    def construct_tree_from_json(self, json_path):
        """
            Construct logic tree from json file.
        :param json_path: str
        :return:
        """
        with open(json_path, 'r', encoding='utf8') as fin:
            nested_dict = json.load(fin)
            if len(nested_dict.keys()) != 1:
                raise ValueError("The number of root key in nested dictionary must be one!")
            root_key = list(nested_dict.keys())[0]
            if '--' in root_key:
                logic_type = root_key.split('--', maxsplit=1)[0]
                sentence = root_key.split('--', maxsplit=1)[-1]
            else:
                logic_type = root_key.split('--', maxsplit=1)[0]
                sentence = None
            root_node = LogicNode(is_root=True, logic_type=logic_type, sentence=sentence)

            self._construct_tree_from_dict(list(nested_dict.values())[0], root_node)
            return root_node

    def _construct_tree_from_dict(self, nested_dict, parent_node):
        """
            Construct logic tree from nested dictionary.
        :param nested_dict: dict
        :param parent_node: LogicNode
        :return:
        """
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                node_key = list(value.keys())[0]
                if '--' in node_key:
                    logic_type = node_key.split('--', maxsplit=1)[0]
                    sentence = node_key.split('--', maxsplit=1)[1]
                else:
                    logic_type = node_key.split('--', maxsplit=1)[0]
                    sentence = None
                node = LogicNode(is_root=True, logic_type=logic_type, sentence=sentence)
                self._construct_tree_from_dict(value[node_key], node)
            else:
                node = LogicNode(is_root=False, logic_type=None, sentence=value)
            parent_node.add_node(node, key)

    def convert_to_nested_dict(self):
        """
            Convert tree to a nested dictionary.
        :return:
        """
        return self.root.convert_to_nested_dict()

    def save_as_png(self, output_dir, filename):
        plot_tree(self.convert_to_nested_dict(), output_dir=output_dir, name=filename)


def convert_json_to_png(json_dir):
    for file in os.listdir(json_dir):
        if not file.endswith('.json'):
            continue
        filepath = os.path.join(json_dir, file)
        tree = LogicTree(root=None, json_path=filepath)
        tree.save_as_png(json_dir, file)


if __name__ == '__main__':
    tree = LogicTree(root=None, json_path=r'C:\NLP-Github\AIPolicy\output\article_parsing\raw-policy\example\22m.json')
    tree.save_as_png(output_dir=r'C:\NLP-Github\AIPolicy\output\article_parsing\raw-policy\example',
                     filename='22m.txt')
