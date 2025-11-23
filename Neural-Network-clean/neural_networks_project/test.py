"""
Visualize a fully connected network as a graph with edges = weights.

- You pass a list of numpy arrays `weights`, where weights[l] has shape
  (n_in, n_out) for layer l (from layer l -> l+1).

- It builds a NetworkX DiGraph and plots it with nodes in vertical columns
  (one column per layer) and edge thickness/color based on weight value.
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


def infer_layer_sizes(weights):
    """
    Infer layer sizes from a list of weight matrices.
    weights[l]: shape (n_in, n_out)
    Returns: list like [n_input, n_hidden1, ..., n_output]
    """
    layer_sizes = []
    for i, W in enumerate(weights):
        n_in, n_out = W.shape
        if i == 0:
            layer_sizes.append(n_in)
        layer_sizes.append(n_out)
    return layer_sizes


def build_network_graph(
    weights,
    max_neurons_per_layer=None,
    min_abs_weight=None,
):
    """
    Build a directed graph from a list of weight matrices.

    Parameters
    ----------
    weights : list of np.ndarray
        Each W has shape (n_in, n_out).
    max_neurons_per_layer : int or None
        If not None, only keep the first `max_neurons_per_layer` neurons
        of each layer (useful for huge networks).
    min_abs_weight : float or None
        If not None, only draw edges with |w| >= min_abs_weight.

    Returns
    -------
    G : networkx.DiGraph
    pos : dict
        Positions for nodes (for plotting).
    layer_sizes : list[int]
        Sizes of (possibly truncated) layers.
    """

    layer_sizes_full = infer_layer_sizes(weights)

    # Optionally truncate neurons per layer
    if max_neurons_per_layer is not None:
        layer_sizes = [min(s, max_neurons_per_layer) for s in layer_sizes_full]
    else:
        layer_sizes = layer_sizes_full

    G = nx.DiGraph()

    # Assign a node id for each neuron in each layer
    # We'll use names like "L0_N0", "L1_N10", etc.
    node_ids = []  # node_ids[layer][i] = node name of neuron i in that layer
    for layer_idx, size in enumerate(layer_sizes):
        ids = []
        for i in range(size):
            node_name = f"L{layer_idx}_N{i}"
            G.add_node(node_name, layer=layer_idx, index=i)
            ids.append(node_name)
        node_ids.append(ids)

    # Add edges with weights
    for l, W in enumerate(weights):
        n_in, n_out = W.shape

        # handle truncation for large networks
        max_in = layer_sizes[l]
        max_out = layer_sizes[l + 1]

        W_view = W[:max_in, :max_out]

        for i in range(max_in):
            for j in range(max_out):
                w = W_view[i, j]
                if min_abs_weight is not None and abs(w) < min_abs_weight:
                    continue
                src = node_ids[l][i]
                dst = node_ids[l + 1][j]
                G.add_edge(src, dst, weight=float(w))

    # Compute node positions (layers on x-axis, neurons spread on y-axis)
    pos = {}
    for layer_idx, ids in enumerate(node_ids):
        n = len(ids)
        # spread vertically between 0 and 1
        if n == 1:
            y_coords = [0.5]
        else:
            y_coords = np.linspace(0.0, 1.0, n)
        x = float(layer_idx)
        for i, node in enumerate(ids):
            pos[node] = (x, y_coords[i])

    return G, pos, layer_sizes




def plot_network_graph(
    G,
    pos,
    figsize=(10, 6),
    small_node_size=5,
    big_node_size=200,
    node_color="#222222",
    cmap_positive="tab:blue",
    cmap_negative="tab:red",
    max_edge_width=0.03,      # base thickness
    min_edge_width=0.003,     # never go thinner than this
    layer_boosts=None,        # <<< per-layer boost here
    filename=None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    plt.figure(figsize=figsize)

    edges = list(G.edges(data=True))
    if not edges:
        print("Graph has no edges to draw.")
        return

    # --- layer info for nodes ---
    node_layers = {n: data["layer"] for n, data in G.nodes(data=True)}
    layers = list(node_layers.values())
    last_layer = max(layers)
    num_layers = last_layer + 1

    # default: no extra boost (all 1.0)
    if layer_boosts is None:
        layer_boosts = [1.0] * num_layers
    else:
        # if you pass a shorter list, pad with 1.0
        if len(layer_boosts) < num_layers:
            layer_boosts = list(layer_boosts) + [1.0] * (num_layers - len(layer_boosts))

    # --- edge widths/colors (global normalization, per-layer boost) ---
    weights_abs = np.array([abs(d["weight"]) for (_, _, d) in edges])
    w_max = weights_abs.max() if len(weights_abs) > 0 else 1.0

    widths = []
    colors = []
    for (u, v, d), w_abs in zip(edges, weights_abs):
        base = (w_abs / w_max) * max_edge_width

        layer_to = node_layers[v]
        boost = layer_boosts[layer_to]

        width = max(base * boost, min_edge_width)
        widths.append(width)
        colors.append(cmap_positive if d["weight"] >= 0 else cmap_negative)

    # --- node sizes (last layer big) ---
    node_sizes = []
    for n, data in G.nodes(data=True):
        if data["layer"] == last_layer:
            node_sizes.append(big_node_size)
        else:
            node_sizes.append(small_node_size)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for (u, v, _) in edges],
        width=widths,
        edge_color=colors,
        alpha=0.7,
        arrows=False,
    )

    # layer labels
    for l in sorted(set(layers)):
        xs = [pos[n][0] for n in G.nodes if node_layers[n] == l]
        ys = [pos[n][1] for n in G.nodes if node_layers[n] == l]
        if xs and ys:
            x_mean = np.mean(xs)
            y_max = max(ys)
            plt.text(x_mean, y_max + 0.07, f"Layer {l}", ha="center", va="bottom", fontsize=9)

    plt.axis("off")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


W1 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer1_season{70}.npy")
W2 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer2_season{70}.npy")
W3 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer3_season{70}.npy")
W4 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer4_season{70}.npy")
W5 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer5_season{70}.npy")
# load your weights somehow
# e.g. if you save them as .npy:

weights = [W1, W2, W3, W4, W5]

G, pos, layer_sizes = build_network_graph(
    weights,
    max_neurons_per_layer=3072,   # good idea for 3072 inputs
    min_abs_weight=0.005         # hide very small weights
)
layer_boosts = [1, 0.04, 0.1, 0.9, 2, 30]
plot_network_graph(
    G,
    pos,
    small_node_size=0.25,
    big_node_size=100,
    max_edge_width=0.02,
    layer_boosts=layer_boosts,      # bump this up if you want even fatter edges
    filename="C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/results.png"
)

