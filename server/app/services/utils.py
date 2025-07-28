import networkx as nx

def build_execution_graph(nodes, edges):
    G = nx.DiGraph()
    node_map = {node['id']: node for node in nodes}

    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    return G, node_map

def get_execution_order(G):
    return list(nx.topological_sort(G))

def is_parallel_node(G, node_id):
    # Returns True if node has multiple incoming edges
    return len(list(G.predecessors(node_id))) > 1
