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


def draw_ciss_graph_web_with_legend(
    G: nx.Graph,
    name: str,
    out_dir: str = "Visualisations",
    height: str = "1200px",
    width: str = "100%",
):
    """
    Interactive web visualization (pyvis) for your CISS schema graph.

    Requirements from user:
    - Do NOT show edge property text on edges
    - Use 2 different edge styles for:
        1) is_column_of
        2) is_key_identifier_of
    - Add a legend in the graph to explain the mapping
    """

    directed = isinstance(G, nx.DiGraph)

    net = Network(
        directed=directed,
        height=height,
        width=width,
        bgcolor="#ffffff",
        font_color="black"
    )
    net.toggle_physics(True)

    # Better interaction
    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 250},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        }
      }
    }
    """)

    # -----------------------------
    # Node styling
    # -----------------------------
    def node_style(attrs: dict):
        t = attrs.get("type", "unknown")
        if t == "table":
            return {"shape": "box", "size": 34, "color": "#4A90E2"}
        if t in ("common column", "common_column"):
            return {"shape": "ellipse", "size": 24, "color": "#F5A623"}
        if t == "column":
            return {"shape": "dot", "size": 14, "color": "#B8B8B8"}
        return {"shape": "dot", "size": 12, "color": "#D0021B"}

    # -----------------------------
    # Edge styling (2 types)
    # -----------------------------
    EDGE_STYLES = {
        "is_column_of": {
            "color": "#9B9B9B",
            "width": 1.2,
            "dashes": False
        },
        "is_key_identifier_of": {
            "color": "#F5A623",
            "width": 3.2,
            "dashes": True
        }
    }

    def get_edge_type(data: dict) -> str:
        # your graph uses property='is_column_of' ... OR edge_type
        return data.get("edge_type") or data.get("property") or data.get("relation") or "is_column_of"

    # -----------------------------
    # Add nodes
    # -----------------------------
    for node, attrs in G.nodes(data=True):
        s = node_style(attrs)

        # Hover tooltip
        node_type = attrs.get("type", "unknown")
        title_parts = [f"<b>{node}</b>", f"type: {node_type}"]

        if attrs.get("table"):
            title_parts.append(f"table: {attrs.get('table')}")
        if attrs.get("column"):
            title_parts.append(f"column: {attrs.get('column')}")
        if attrs.get("key_identifier") is not None:
            title_parts.append(f"key_identifier: {attrs.get('key_identifier')}")

        desc = attrs.get("description")
        if desc:
            short_desc = (desc[:400] + "...") if len(desc) > 400 else desc
            title_parts.append(f"<br><b>description</b>: {short_desc}")

        net.add_node(
            str(node),
            label=str(node),
            title="<br>".join(title_parts),
            color=s["color"],
            size=s["size"],
            shape=s["shape"]
        )

    # -----------------------------
    # Add edges (NO label text)
    # -----------------------------
    for u, v, data in G.edges(data=True):
        e_type = get_edge_type(data)
        style = EDGE_STYLES.get(e_type, {"color": "#7F8C8D", "width": 1.0, "dashes": False})

        net.add_edge(
            str(u),
            str(v),
            # No label here -> no edge text shown
            title=f"type: {e_type}",
            color=style["color"],
            width=style["width"],
            dashes=style["dashes"]
        )

    # -----------------------------
    # Add Legend (fixed position, no physics)
    # -----------------------------
    # Place legend in top-left area using fixed coordinates.
    # These coordinates work well for most graphs; you can tweak if needed.
    legend_title_id = "__LEGEND_TITLE__"
    legend_a1 = "__LEGEND_A1__"
    legend_a2 = "__LEGEND_A2__"
    legend_b1 = "__LEGEND_B1__"
    legend_b2 = "__LEGEND_B2__"

    # Turn off physics for legend nodes by setting fixed x,y + physics=False
    net.add_node(
        legend_title_id,
        label="Legend",
        shape="box",
        color="#FFFFFF",
        font={"size": 18, "face": "arial", "bold": True},
        x=-1100, y=-700, fixed=True, physics=False
    )

    # Legend row 1: is_column_of
    net.add_node(legend_a1, label="", shape="dot", color="#FFFFFF",
                 x=-1100, y=-620, fixed=True, physics=False)
    net.add_node(legend_a2, label="is_column_of", shape="box", color="#FFFFFF",
                 x=-980, y=-620, fixed=True, physics=False)

    style1 = EDGE_STYLES["is_column_of"]
    net.add_edge(
        legend_a1, legend_a2,
        color=style1["color"],
        width=style1["width"],
        dashes=style1["dashes"],
        physics=False
    )

    # Legend row 2: is_key_identifier_of
    net.add_node(legend_b1, label="", shape="dot", color="#FFFFFF",
                 x=-1100, y=-540, fixed=True, physics=False)
    net.add_node(legend_b2, label="is_key_identifier_of", shape="box", color="#FFFFFF",
                 x=-980, y=-540, fixed=True, physics=False)

    style2 = EDGE_STYLES["is_key_identifier_of"]
    net.add_edge(
        legend_b1, legend_b2,
        color=style2["color"],
        width=style2["width"],
        dashes=style2["dashes"],
        physics=False
    )

    # -----------------------------
    # Output HTML
    # -----------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(os.getcwd(), out_dir, f"{name}_graph_visualisation.html")
    net.show(out_path, notebook=False)
    print(f"Saved interactive graph to: {out_path}")
    return out_path

