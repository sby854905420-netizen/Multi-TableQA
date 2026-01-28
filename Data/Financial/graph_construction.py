import networkx as nx

def build_variable_dependency_graph():
    G = nx.DiGraph()
    disp_node = ("disp_id", "disp")
    for var in ["client_id","type"]:
        G.add_edge((var, "disp"), disp_node)
    G.add_edge(disp_node, ("account_id", "disp"))
        
    account_node = ("account_id", "account")
    for var in ["district_id", "frequency","date"]:
        G.add_edge((var, "account"), account_node)
        
    card_node = ("card_id", "card")
    for var in [ "type","issued"]:
        G.add_edge((var, "card"), card_node)
    
    client_node = ("client_id", "client")
    for var in ["gender", "birth_date","district_id"]:
        G.add_edge((var, "client"), client_node)   
        
    district_node = ("district_id", "district")
    for var in ["A2", "A3","A4","A5", "A6","A7","A8", "A9","A10","A11", "A12","A13","A14", "A15","A16"]:
        G.add_edge((var, "district"), district_node) 
        
    loan_node = ("loan_id", "loan")
    for var in [ "date","amount","duration", "payments","status"]:
        G.add_edge((var, "loan"), loan_node) 
    G.add_edge(loan_node, ("account_id", "loan"))
        
    order_node = ("order_id", "order")
    for var in ["bank_to","account_to","amount", "k_symbol"]:
        G.add_edge((var, "order"), order_node) 
    G.add_edge(order_node, ("account_id", "order"))

    trans_node = ("trans_id", "trans")
    for var in [ "date","type","operation", "amount","balance","bank","account","k_symbol"]:
        G.add_edge((var, "trans"), trans_node) 
    G.add_edge(trans_node, ("account_id", "trans"))

    G.add_edge(  ("account_id", "trans"),account_node)
    G.add_edge( ("account_id", "order"),account_node)
    G.add_edge(  ("account_id", "loan"), account_node) 
    G.add_edge(  ("account_id", "disp"),account_node)
    G.add_edge( client_node, ("card_id","disp"))
    G.add_edge( district_node, ("district_id","client"))
    G.add_edge( disp_node, ("disp_id","card"))
    G.add_edge( district_node, ("district_id","account"))

    return G


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
    
    

