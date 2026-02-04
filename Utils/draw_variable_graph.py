from pyvis.network import Network
import networkx as nx
import os

def draw_graph(G:nx.DiGraph,name:str):
    net = Network(
        directed=True,
        height="1200px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black"
    )

    for node in G.nodes():
        label = node[0]
        title = f"Variable: {node[0]}<br>Table: {node[1]}"
        net.add_node(str(node), label=label, title=title)

    for u, v, data in G.edges(data=True):
        net.add_edge(
            str(u), str(v),
            dashes=data.get("weight") == 0.3,
            title=f"weight={data.get('weight', 1.0)}"
        )

    basic_path = os.getcwd()
    net.show(f"{basic_path}/Visualisations/{name}_graph_visualisation.html", notebook=False)


