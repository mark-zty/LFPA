import numpy as np
from tqdm import trange

class model_ar:
    """
    Creates a preferential attachment graph object for the alpha variant of the k model, with the option for changing the redirection probability r.

    Attributes
    ----------
    N: integer
        Number of initial nodes in the network
    edge_list: list of lists
        Adjacency list for direct first degree connections in the network (corresponding to observed network)
    deg_dist: list
        Degree of node i at index i.
    weights: list
        Weight of node i at index i
    targets: list of integers
        Weighted selection list of nodes corresponding to weight k^alpha
    T: integer
        Length of targets
    r: float
        Redirection probability
        
    Methods
    -------
    increase_size(size)
        Increase the number of nodes to "size" via preferential attachment rules
    """

    def __init__(self, initial_el=[[0,1], [1,2]], size=3, initial_weights=[False], alpha = 0.0):
        self.id = id
        self.edge_list = initial_el
        self.targets = []
        self.N = size
        self.deg_dist = [0 for x in range(self.N)] # initial degree distribution, format: degree of node i at index i   
        self.weights = [0 for x in range(self.N)] # initial weights, format: weight of node i at index i
        self.T = 0 # length of targets
        self.neighbors = [[] for x in range(self.N)] # adjacency list
        for i in range(len(self.edge_list)):
            first = self.edge_list[i][0]
            second = self.edge_list[i][1]
            if first != second: # not a self-loop
                self.neighbors[first].append(second)
                self.neighbors[second].append(first)
                self.deg_dist[first] += 1
                self.deg_dist[second] += 1
            else: # self-loop
                self.neighbors[first].append(first)
                self.deg_dist[first] += 1
        # Efficient max/min degree sets for inf/-inf
        self.max_deg = max(self.deg_dist) if self.N > 0 else 0
        self.min_deg = min(self.deg_dist) if self.N > 0 else 0
        self.max_deg_nodes = set([i for i, d in enumerate(self.deg_dist) if d == self.max_deg])
        self.min_deg_nodes = set([i for i, d in enumerate(self.deg_dist) if d == self.min_deg])
        if not any(initial_weights): # no initial weights, weights determined by k^alpha
            if alpha == 1:
                for i in range(self.N):
                    for j in range(self.deg_dist[i]):
                        self.targets.append(i) # append node i to targets
                        self.T += 1 # increment length of targets
                    self.weights[i] = self.deg_dist[i] # weight of node i is its degree
            else:
                self.small_num = 10
                self.smalls = [[] for x in range(self.small_num)]
                self.len_smalls = [0 for x in range(self.small_num)]
                self.total_weight = 0.0
                self.ind_to_weight = {} # index to weight mapping
                self.weight_to_ind = {} # weight to index mapping
                self.weights = []
                for i in range(self.N):
                    degree = self.deg_dist[i]
                    if degree <= self.small_num:
                        # Cast degree^alpha to float
                        weight = float(degree ** alpha)
                        self.smalls[degree - 1].append(i)
                        self.len_smalls[degree - 1] += 1
                        self.total_weight += weight
                    else:
                        # Cast degree^alpha to float
                        weight = float(degree ** alpha)
                        self.ind_to_weight[i] = len(self.weights)
                        self.weight_to_ind[len(self.weights)] = i
                        self.weights.append(weight)
                        self.total_weight += weight
        else: # if initial weights are provided
            self.weights = initial_weights
            self.T += np.sum(initial_weights)
            for i in range(len(self.weights)):
                counter = self.weights[i]
                while counter > 0:
                    self.targets.append(i)
                    counter -= 1
        self.alpha = alpha

    def increase_size(self, size,r:float=1.0, show_progress=True):
        range_func = trange if show_progress else range
        if self.alpha == 0:  # Targets remain the same (random attachment)
            for i in range_func(size - self.N):
                target_node = np.random.randint(0, self.N)
                if np.random.rand() < r:
                    new = self.neighbors[target_node][np.random.randint(0, len(self.neighbors[target_node]))]
                else:
                    new = target_node
                self.edge_list.append([new, self.N])
                self.deg_dist[new] += 1
                self.deg_dist.append(1)
                self.neighbors[new].append(self.N)
                self.neighbors.append([new])
                self.N += 1
        elif self.alpha == 1:  # Preferential attachment (degree-proportional weights)
            for i in range_func(size - self.N):
                target_node = self.targets[np.random.randint(0, self.T)]
                if np.random.rand() < r: 
                    new = self.neighbors[target_node][np.random.randint(0, len(self.neighbors[target_node]))]
                else:
                    new = target_node
                self.edge_list.append([new, self.N])
                self.deg_dist[new] += 1
                self.deg_dist.append(1)
                self.neighbors[new].append(self.N)
                self.neighbors.append([new])
                self.targets.append(new)
                self.targets.append(self.N)
                self.weights[new] += 1
                self.weights.append(1)
                self.T += 2
                self.N += 1
        elif self.alpha is not None and (self.alpha == np.inf or self.alpha == float('inf')):
            for i in range_func(size - self.N):
                # Efficiently choose node with largest degree
                target_node = np.random.choice(list(self.max_deg_nodes))
                d = self.deg_dist[target_node]
                r_value = r
                if np.random.rand() < r_value:
                    chosen_node = self.neighbors[target_node][np.random.randint(0, len(self.neighbors[target_node]))]
                else:
                    chosen_node = target_node
                self.edge_list.append([chosen_node, self.N])
                # Update degree and neighbors
                self.deg_dist[chosen_node] += 1
                self.deg_dist.append(1)
                self.neighbors[chosen_node].append(self.N)
                self.neighbors.append([chosen_node])
                # Update max_deg and max_deg_nodes
                if self.deg_dist[chosen_node] > self.max_deg:
                    self.max_deg = self.deg_dist[chosen_node]
                    self.max_deg_nodes = {chosen_node}
                elif self.deg_dist[chosen_node] == self.max_deg:
                    self.max_deg_nodes.add(chosen_node)
                else:
                    prev_max = self.max_deg - 1
                    if prev_max >= 0 and chosen_node in self.max_deg_nodes:
                        self.max_deg_nodes.remove(chosen_node)
                if 1 > self.max_deg:
                    self.max_deg = 1
                    self.max_deg_nodes = {self.N}
                elif 1 == self.max_deg:
                    self.max_deg_nodes.add(self.N)
                self.N += 1
        elif self.alpha is not None and (self.alpha == -np.inf or self.alpha == float('-inf')):
            for i in range_func(size - self.N):
                # Efficiently choose node with smallest degree
                target_node = np.random.choice(list(self.min_deg_nodes))
                d = self.deg_dist[target_node]
                r_value = r
                if np.random.rand() < r_value:
                    chosen_node = self.neighbors[target_node][np.random.randint(0, len(self.neighbors[target_node]))]
                else:
                    chosen_node = target_node
                self.edge_list.append([chosen_node, self.N])
                # Update degree and neighbors
                self.deg_dist[chosen_node] += 1
                self.deg_dist.append(1)
                self.neighbors[chosen_node].append(self.N)
                self.neighbors.append([chosen_node])
                # Update min_deg and min_deg_nodes
                if self.deg_dist[chosen_node] < self.min_deg:
                    self.min_deg = self.deg_dist[chosen_node]
                    self.min_deg_nodes = {chosen_node}
                elif self.deg_dist[chosen_node] == self.min_deg:
                    self.min_deg_nodes.add(chosen_node)
                else:
                    prev_min = self.min_deg + 1
                    if prev_min >= 0 and chosen_node in self.min_deg_nodes:
                        self.min_deg_nodes.remove(chosen_node)
                if 1 < self.min_deg:
                    self.min_deg = 1
                    self.min_deg_nodes = {self.N}
                elif 1 == self.min_deg:
                    self.min_deg_nodes.add(self.N)
                self.N += 1
        else:  # Generic alpha case (optimized for small/large degrees)
            for i in range_func(size - self.N):
                # Select target_node based on weights
                ratios = np.zeros(self.small_num)
                for j in range(self.small_num):
                    for k in range(j, self.small_num):
                        ratios[k] += (self.len_smalls[j] * (j + 1)**self.alpha) / self.total_weight
                rand_val = np.random.rand()
                cutoff = 0
                while cutoff < self.small_num and rand_val > ratios[cutoff]:
                    cutoff += 1
                if cutoff < self.small_num:
                    target_node = self.smalls[cutoff][np.random.randint(0, self.len_smalls[cutoff])]
                    if np.random.rand() < r:
                        chosen_node = self.neighbors[target_node][np.random.randint(0, len(self.neighbors[target_node]))]
                    else:
                        chosen_node = target_node
                else:
                    # Ensure weights are floats and avoid division issues
                    weights_array = np.array(self.weights, dtype=np.float64)
                    total_weight = np.sum(weights_array)
                    if total_weight == 0:
                        prob = np.ones_like(weights_array) / len(weights_array)
                    else:
                        prob = weights_array / total_weight
                    # Use float-safe probabilities
                    rand_weight = np.random.choice(len(self.weights), p=prob)
                    target_node = self.weight_to_ind[rand_weight]
                    if np.random.rand() < r:
                        chosen_node = self.neighbors[target_node][np.random.randint(0, len(self.neighbors[target_node]))]
                    else:
                        chosen_node = target_node

                # Update data structures for chosen_node
                if self.deg_dist[chosen_node] <= self.small_num - 1:
                    self.smalls[self.deg_dist[chosen_node]].append(chosen_node)
                    self.len_smalls[self.deg_dist[chosen_node]] += 1
                    self.smalls[self.deg_dist[chosen_node] - 1].remove(chosen_node)
                    self.len_smalls[self.deg_dist[chosen_node] - 1] -= 1
                    self.total_weight += (self.deg_dist[chosen_node] + 1)**self.alpha - self.deg_dist[chosen_node]**self.alpha
                elif self.deg_dist[chosen_node] == self.small_num:
                    self.ind_to_weight[chosen_node] = len(self.weights)
                    self.weight_to_ind[len(self.weights)] = chosen_node
                    self.weights.append((self.small_num + 1)**self.alpha)
                    self.smalls[self.small_num - 1].remove(chosen_node)
                    self.len_smalls[self.small_num - 1] -= 1
                    self.total_weight += (self.deg_dist[chosen_node] + 1)**self.alpha - self.deg_dist[chosen_node]**self.alpha
                else:
                    added_weight = (self.deg_dist[chosen_node] + 1)**self.alpha - self.weights[self.ind_to_weight[chosen_node]]
                    self.weights[self.ind_to_weight[chosen_node]] += added_weight
                    self.total_weight += added_weight

                # Add new node to the "degree=1" group
                self.smalls[0].append(self.N)
                self.len_smalls[0] += 1
                self.total_weight += 1

                # Update edges and degrees
                self.edge_list.append([chosen_node, self.N])
                self.deg_dist[chosen_node] += 1
                self.deg_dist.append(1)
                self.neighbors[chosen_node].append(self.N)
                self.neighbors.append([chosen_node])
                self.N += 1