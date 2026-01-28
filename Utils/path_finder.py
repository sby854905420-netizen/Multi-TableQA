import networkx as nx




def find_all_paths(G,start_node,target_node):

    if G.has_node(start_node) and not G.has_node(target_node):

        all_paths = list(nx.all_simple_paths(G, source=start_node, target=target_node))

        if len(all_paths) > 0:
            print(f"Total paths found: {len(all_paths)}")  
            return all_paths
        else:
            print("No path found!")
            return None
    else:
        if not G.has_node(start_node):
            print("The graph does not contain the start node!")
        if not G.has_node(target_node):
            print("The graph does not contain the target node!")
        return None