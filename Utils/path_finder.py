import networkx as nx
import pandas as pd
from collections.abc import Iterable

def find_all_paths(G,start_node,target_node):

    if G.has_node(start_node) and G.has_node(target_node):
        all_paths = list(nx.all_simple_paths(G, source=start_node, target=target_node))

        if len(all_paths) > 0:
            # print(f"Total paths found: {len(all_paths)}")  
            return all_paths
        else:
            # print(f"No path found from {start_node} to {target_node}")
            return []
    else:
        if not G.has_node(start_node):
            print("The graph does not contain the start node!")
        if not G.has_node(target_node):
            print("The graph does not contain the target node!")
        return []
    