import networkx as nx
from random import shuffle
from scipy.spatial.distance import cosine as dcos
import matplotlib.pyplot as plt
from utils import timing


class Cluster:

    def __init__(self, thresh = 0.3):
        self.G = nx.Graph()
        self.names = []
        self.class_idx = 0
        self.node_idx = 0
        self.threshold = thresh
        self.people_idx = {}
        self.max_faces = 100

# ===================================== GENERAL UTILITIES =========================================

    def adjust_indexes(self):
        """
        This function restores the continuity of the nodes in the graph, after
        one or more has been deleted.
        :return: none
        """
        i = 0
        nodes = []
        for node in self.G.nodes:
            if node != i:
                nodes.append((i, node))
            i += 1

        # restore nodes
        for node in nodes:
            self.G.add_node(node[0], name = self.G.node[node[1]]['name'], desc = self.G.node[node[1]]['desc'])
            new_edges = []
            # restore edges
            for edge in self.G.edges:
                if edge[0] == node[1]:
                    new_edges.append((node[0], edge[1], {'weight': self.G[edge[0]][edge[1]]['weight']}))
                elif edge[1] == node[1]:
                    new_edges.append((edge[0], node[0], {'weight': self.G[edge[0]][edge[1]]['weight']}))
            self.G.add_edges_from(new_edges)
            self.G.remove_node(node[1])

    def check_index(self, newname):
        """
        This function restores the continuity of the identities, in case of a new
        identity been added to the list it assigns the corresponding face count,
        otherwise it just moves back the indexes
        :param newname: bool to notify if the last name in name list is of
        recent addition
        :return: none
        """
        if newname:
            try:
                self.people_idx[self.names[-1]] = self.people_idx.pop(0)
            except KeyError:
                pass

        new_indexes = []
        for key in self.people_idx.items():
            if isinstance(key[0], int):
                new_indexes.append(key)
        for indexes in new_indexes:
            if indexes[0] != 0 and indexes[0] - 1 not in self.people_idx:
                self.people_idx.pop(indexes[0])
                self.people_idx[indexes[0]-1] = indexes[1]

    def clear_idx(self, idx):
        """
        This function selects all nodes of a given index for deletion, adjusting
        at the same time the index of the other nodes
        :param idx: the index to remove
        :return: a list of nodes of the given index
        """
        node_to_remove = []
        for i in range(self.node_idx):
            if self.G.node[i]['name'] == idx:
                node_to_remove.append(i)
            elif isinstance(idx, int):
                if isinstance(self.G.node[i]['name'], int):
                    try:
                        if self.G.node[i]['name'] > idx:
                            self.G.node[i]['name'] = int(self.G.node[i]['name']) - 1
                    except ValueError:
                        pass
        return node_to_remove

    def clear_class(self, idx):
        """
        This function deletes a class from the dictionary, along with all the
        corresponding nodes in the graph.
        :param idx: index or name of the class to delete
        :return: none
        """
        nodes_to_remove = self.clear_idx(idx)

        for node in nodes_to_remove:
            self.G.remove_node(node)
        try:
            del self.people_idx[idx]
        except KeyError:
            pass
        self.check_index(False)
        self.node_idx = len(self.G.nodes.data())

        self.adjust_indexes()

        if isinstance(idx, int):
            self.class_idx -= 1

# ============================= COMPLETE CHINESE WHISPERS UTILITIES ===============================

# removal of nodes or edges
    def check_consistency(self, node, length):
        if length < self.people_idx[self.G.node[node]['name']] / 5:
            self.people_idx[self.G.node[node]['name']] -= 1
            self.G.remove_node(node)
            self.adjust_indexes()
            self.node_idx -= 1
            return False
        else:
            return True

    def clear_wrong_neighs(self, node, neighs):
        edges_to_remove = []
        for ne in neighs:
            if self.G.node[node]['name'] != self.G.node[ne]['name']:
                edges_to_remove.append((node, ne))

        for i in range(len(edges_to_remove)):
            self.G.remove_edge(edges_to_remove[i][0], edges_to_remove[i][1])

# ========================== Functions to handle people != subgraphs ==============================

# more identities than subgraphs
    def choose_indexes(self, faces):
        components = list(nx.connected_components(self.G))
        i = 0
        j = 0
        while i < len(components) and len(components[i]) == faces[i]:
            i += 1

        if i < len(components):
            if i == 0:
                j = 1

            while j < len(faces) and len(components[i]) - faces[i] != faces[j]:
                j += 1
                if j == i:
                    j += 1
            return i, j
        else:
            return -1, -1

    def select_excess_nodes(self, name):
        nodes_to_remove = []

        for k in range(self.node_idx):
            if self.G.node[k]['name'] == name:
                nodes_to_remove.append(k)

        return nodes_to_remove

    def select_excess_class(self, faces, i, j):
        classes = list(self.people_idx.keys())

        if j < len(faces):
            if (isinstance(classes[i], str) and isinstance(classes[j], str)) or \
                    (not isinstance(classes[i], str) and not isinstance(classes[j], str)):
                if faces[i] > faces[j]:
                    nodes_to_remove = self.select_excess_nodes(classes[j])
                else:
                    nodes_to_remove = self.select_excess_nodes(classes[i])
            elif isinstance(classes[i], str):
                nodes_to_remove = self.select_excess_nodes(classes[j])
            else:
                nodes_to_remove = self.select_excess_nodes(classes[i])
        else:
            nodes_to_remove = self.select_excess_nodes(classes[i])

        return nodes_to_remove

    def delete_excess_class(self):
        faces = list(self.people_idx.values())

        i, j = self.choose_indexes(faces)

        if i > 0:
            nodes_to_remove = self.select_excess_class(faces, i, j)
            for node in nodes_to_remove:
                self.people_idx[self.G.node[node]['name']] -= 1
                self.G.remove_node(node)

            self.adjust_indexes()
            self.node_idx = len(self.G.nodes.data())

# more subgraphs than identities
    def delete_subgraph(self, nh):
        subgraph_len_min = self.max_faces
        component_to_delete = 0
        while len(self.people_idx) < nx.number_connected_components(self.G):
            for nodes in nh:
                if len(nx.node_connected_component(self.G, nodes)) < subgraph_len_min:
                    subgraph_len_min = len(nx.node_connected_component(self.G, nodes))
                    component_to_delete = nodes

            for node in nx.node_connected_component(self.G, component_to_delete):
                try:
                    nh.remove(node)
                except ValueError:
                    pass
                self.people_idx[self.G.node[node]['name']] -= 1
                self.G.node[node]['name'] = 'delete'

            self.clear_class('delete')

    def check_subgraphs(self, nh):
        if len(self.people_idx) < nx.number_connected_components(self.G):
            self.delete_subgraph(nh)
        elif len(self.people_idx) > nx.number_connected_components(self.G):
            self.delete_excess_class()
        return True

# ============================ Functions to perform chinese whispers ==============================

    def find_max_class(self, classes, shuffled_nodes, index):
        # find the class with the highest edge weight sum
        max = 0
        maxclass = self.G.node[shuffled_nodes[index]]['name']
        for c in classes:
            if classes[c] > max:
                max = classes[c]
                maxclass = c
        # set the class of target node to the winning local class
        if maxclass != self.G.node[shuffled_nodes[index]]['name']:
            oldclass = self.G.node[shuffled_nodes[index]]['name']
            self.people_idx[oldclass] -= 1
            self.G.node[shuffled_nodes[index]]['name'] = maxclass
            self.people_idx[maxclass] += 1

    def check_classes(self, neighs, shuffled_nodes, index):
        # do an inventory of the given nodes neighbours and edge weights
        classes = {}
        for ne in neighs:
            if isinstance(ne, int):
                if self.G.node[ne]['name'] in classes:
                    classes[self.G.node[ne]['name']] += self.G[shuffled_nodes[index]][ne]['weight']
                else:
                    classes[self.G.node[ne]['name']] = self.G[shuffled_nodes[index]][ne]['weight']
        return classes

    def perform_chinese_iteration(self, shuffled_nodes):
        i = 0
        g_modified = False
        while i < len(shuffled_nodes) and not g_modified:
            neighs = self.G.adj[shuffled_nodes[i]]
            if self.check_consistency(shuffled_nodes[i], len(neighs)):

                classes = self.check_classes(neighs, shuffled_nodes, i)

                self.find_max_class(classes, shuffled_nodes, i)

                if self.check_consistency(shuffled_nodes[i], len(neighs)):
                    self.clear_wrong_neighs(shuffled_nodes[i], neighs)
                else:
                    print(list(neighs))
                    print(shuffled_nodes[i])
                    nh = list(neighs)
                    for j in range(len(nh)):
                        if nh[j] > shuffled_nodes[i]:
                            nh[j] -= 1

                    print(nh)
                    g_modified = self.check_subgraphs(nh)
            else:
                print(list(neighs))
                print(shuffled_nodes[i])
                nh = list(neighs)
                for j in range(len(nh)):
                    if nh[j] > shuffled_nodes[i]:
                        nh[j] -= 1

                print(nh)
                g_modified = self.check_subgraphs(nh)

            i += 1

    def clear_people(self):
        empty_index = [x for x in self.people_idx if not self.people_idx[x]]

        for person in empty_index:
            self.clear_class(person)

        return empty_index

    @timing
    def chinese_whispers(self):
        iterations = 10
        deleted = []
        for z in range(iterations):
            gn = [i for i in range(self.node_idx)]
            # I randomize the nodes to give me an arbitrary start point
            shuffle(gn)

            self.perform_chinese_iteration(gn)

            empty = self.clear_people()

            for x in empty:
                deleted.append(x)

        return deleted

# ==================================== END OF EXECUTION FUNCTIONS ==================================

    def check_names(self):
        for i in range(self.node_idx):
            idx = self.G.node[i]['name']
            if isinstance(idx, int):
                if idx == 0:
                    self.G.node[i]['name'] = self.names[-1]
                else:
                    self.G.node[i]['name'] -= 1

    def add_name(self, name):
        if name not in self.names:
            self.names.append(name)
            self.check_index(True)
            self.check_names()
            self.class_idx -= 1
        else:
            print('Name already in list, use a new one')

    def plot_graph(self):
        from utils import colors

        pos = nx.spring_layout(self.G)
        color_list = colors(len(self.people_idx))
        plt.title("Connected components in the Chinese Whispers Graph")
        wcc = nx.connected_component_subgraphs(self.G)
        lab = ["Person {}".format(x) for x in self.people_idx.keys()]
        for index, sg in enumerate(wcc):
            nx.draw_networkx(sg, pos=pos, edge_color=color_list[index], node_color=color_list[index])
        for i in range(len(lab)):
            plt.scatter([], [], c=color_list[i], alpha=0.3, s=100, label=lab[i])
        plt.legend(title="Clusters")
        plt.show()

# ================================= COMMON EXECUTION FUNCTIONS ====================================

    def check_distances(self):
        for i in range(self.node_idx):
            distance = dcos(self.G.node[i]['desc'], self.G.node[self.node_idx]['desc'])
            # distance = abs(self.G.node[i]['desc'] - self.G.node[self.node_idx]['desc'])
            if distance <= self.threshold:
                self.G.add_edge(i, self.node_idx, weight = distance)

    def get_distance(self, node0, node1):
        return dcos(self.G.node[node0]['desc'], self.G.node[node1]['desc'])

    def clear_old(self, class_idx):
        i = 0
        while self.G.node[i]['name'] != class_idx:
            i += 1
        self.G.remove_node(i)
        self.people_idx[class_idx] -= 1
        self.adjust_indexes()
        self.node_idx -= 1

    def update_graph(self, desc):

        self.G.add_node(self.node_idx, name = self.class_idx, desc = desc)
        self.check_distances()
        neighs = self.G.adj[self.node_idx]
        classes = {}
        # do an inventory of the given nodes neighbours and edge weights
        for ne in neighs:
            if isinstance(ne, int):
                identity = self.G.node[ne]['name']
                if identity in classes:
                    classes[identity] += self.G[self.node_idx][ne]['weight']
                else:
                    classes[identity] = self.G[self.node_idx][ne]['weight']
        # find the class with the lowest edge weight sum
        class_idx = self.class_idx
        if classes:
            class_idx = min(classes, key=classes.get)
            self.class_idx -= 1

        self.G.node[self.node_idx]['name'] = class_idx

        if class_idx in self.people_idx.keys():
            self.people_idx[class_idx] += 1
            if self.people_idx[class_idx] > self.max_faces:
                self.clear_old(class_idx)
        else:
            self.people_idx[class_idx] = 1

        self.class_idx += 1
        self.node_idx += 1
