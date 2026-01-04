import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_toric_brickwall_layers(rows, cols):
    if rows % 2 != 0 or cols % 2 != 0:
        print("Visualization requires even dimensions for PBC.")
        return

    num_qubits = rows * cols
    G = nx.Graph()
    G.add_nodes_from(range(num_qubits))

    # Position nodes in a grid (using negative y so row 0 is at top)
    pos = {}
    for r in range(rows):
        for c in range(cols):
            pos[r * cols + c] = np.array([c, -r])

    # Define edges for the 4 layers
    h_even_edges = []
    h_pbc_edges = []
    for r in range(rows):
        for c in range(0, cols, 2):
            u, v = r * cols + c, r * cols + (c + 1) % cols
            if (c + 1) == cols: h_pbc_edges.append((u, v))
            else: h_even_edges.append((u, v))

    h_odd_edges = []
    h_odd_pbc_edges = []
    for r in range(rows):
        for c in range(1, cols, 2):
            u, v = r * cols + c, r * cols + (c + 1) % cols
            if (c + 1) == cols: h_odd_pbc_edges.append((u, v))
            else: h_odd_edges.append((u, v))

    v_even_edges = []
    v_pbc_edges = []
    for c in range(cols):
        for r in range(0, rows, 2):
            u, v = r * cols + c, ((r + 1) % rows) * cols + c
            if (r + 1) == rows: v_pbc_edges.append((u, v))
            else: v_even_edges.append((u, v))

    v_odd_edges = []
    v_odd_pbc_edges = []
    for c in range(cols):
        for r in range(1, rows, 2):
            u, v = r * cols + c, ((r + 1) % rows) * cols + c
            if (r + 1) == rows: v_odd_pbc_edges.append((u, v))
            else: v_odd_edges.append((u, v))

    layers = [
        ("1. Horizontal Even", h_even_edges, h_pbc_edges, 'blue'),
        ("2. Horizontal Odd (w/ Wrap)", h_odd_edges, h_odd_pbc_edges, 'cyan'),
        ("3. Vertical Even", v_even_edges, v_pbc_edges, 'red'),
        ("4. Vertical Odd (w/ Wrap)", v_odd_edges, v_odd_pbc_edges, 'magenta'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, (title, reg_edges, pbc_edges, color) in zip(axes, layers):
        ax.set_title(title)
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=400)
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        # Draw regular edges straight
        nx.draw_networkx_edges(G, pos, edgelist=reg_edges, ax=ax, edge_color=color, width=3)
        
        # Draw PBC edges curved
        for u, v in pbc_edges:
            # Determine curvature direction based on layer type
            rad = 0.3 if 'Horizontal' in title else -0.3
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, 
                                   edge_color=color, width=3, style='dashed',
                                   connectionstyle=f'arc3, rad={rad}')

        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_toric_brickwall_layers(4, 4)