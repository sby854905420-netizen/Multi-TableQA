import networkx as nx
import matplotlib.pyplot as plt


def build_variable_dependency_graph():
    G = nx.DiGraph()


    edges = [
    # (source, target, weight)
    (("noc", "noc_region"), ("id", "noc_region"), 1),
    (("region_name", "noc_region"), ("id", "noc_region"), 1),
    (("id", "noc_region"), ("region_id", "person_region"), 2),  # red
  
    (("id", "person"),("person_id", "person_region"),  2),  # red
    (("full_name", "person"), ("id", "person"), 1),
    (("height", "person"), ("id", "person"), 1),
    (("weight", "person"), ("id", "person"), 1),
    (("gender", "person"), ("id", "person"), 1),
    
    (("id", "person"), ("person_id", "games_competitor"), 2),  # red
    (("id", "games"), ("games_id", "games_competitor"), 2),  # red
    
    (("games_id", "games_city"), ("id", "games"), 2),  # red
    (("id", "city"), ("city_id", "games_city"), 2),  # red
    (("city_name", "city"), ("id", "city"), 1),

    (("games_name", "games"), ("id", "games"), 1),
    (("games_year", "games"), ("id", "games"), 1),
    (("season", "games"), ("id", "games"), 1),
    


    (("age", "games_competitor"), ("id", "games_competitor"), 1),
    (("id", "games_competitor"), ("competitor_id", "competitor_event"), 2),  # red
    (("id", "event"), ("event_id", "competitor_event"), 2),  # red
    (("event_name", "event"), ("id", "event"), 1),
    (("id", "sport"), ("sport_id", "event"), 2),  # red
    (("sport_name", "sport"), ("id", "sport"), 1),


    ( ("id", "medal"), ("medal_id", "competitor_event"),2),  # red
    (("medal_name", "medal"), ("id", "medal"), 1),
    
    (("sport_id", "event"), ("event_id", "event"), 1),
    (("region_id", "person_region"), ("person_id", "person_region"), 1),
    (("city_id", "games_city"), ("games_id", "games_city"), 1), 
    (("games_id", "games_competitor"), ("id", "games_competitor"), 1),
    (("person_id", "games_competitor"), ("id", "games_competitor"), 1),
    ( ("medal_id", "competitor_event"), ("event_id", "competitor_event"), 1),
    ( ("medal_id", "competitor_event"), ("competitor_id", "competitor_event"), 1),
    ( ("event_id", "competitor_event"), ("competitor_id", "competitor_event"),1)]


    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)
    return G


def find_all_paths_to_caseid(start_variable, start_sheet):
    G = build_variable_dependency_graph()
    start_node = (start_variable, start_sheet)
    target_node = ("competitor_id", "competitor_event")
    # target_node = ("Model", "VPICDECODE")
    

    if not G.has_node(start_node) or not G.has_node(target_node):
        return None

    try:
        all_paths = list(nx.all_simple_paths(G, source=start_node, target=target_node))
        # print(f"Total paths found: {len(all_paths)}")

        filtered_paths = []

        for i, path in enumerate(all_paths):
            skip_from_index = 0

           
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                edge_data = G.get_edge_data(u, v)
                if edge_data and edge_data.get("weight") == 0.3:
                    skip_from_index = j + 1 
                    break  

      
            filtered_path = path[skip_from_index:]
            # if filtered_path and filtered_path[-1] == ("CASEID", "center"):
                
            #     filtered_path = filtered_path[:-1]

            if len(filtered_path) > 1:
                filtered_paths.append(filtered_path)

        return filtered_paths

    except nx.NetworkXNoPath:
        print("No path found.")
        return None
    
    
G = build_variable_dependency_graph() 

