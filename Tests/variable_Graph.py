import networkx as nx


#     return G
import networkx as nx

def build_variable_dependency_graph():
    G = nx.DiGraph()  
    p = 0.3


    CASEID_node = ("CASEID", "center")
    G.add_edge(("CASEID", "CRASH"), CASEID_node, weight=0.5)
    G.add_edge( ("CASEID", "GV"),CASEID_node,weight=0.5)
    G.add_edge(("CASEID", "VPICDECODE"),CASEID_node,weight=0.5)
    G.add_edge(("CASEID", "EVENT"), CASEID_node,weight=0.5)
    G.add_edge( ("CASEID", "AVOID"), CASEID_node, weight=0.5)
    G.add_edge( ("CASEID", "OCC"),CASEID_node,weight=0.5)
    G.add_edge( ("CASEID", "OCCONTACT"),CASEID_node,weight=0.5)
    
    G.add_edge(("CATEGORY", "CRASH"), ("SUMMARY", "CRASH"))
    G.add_edge(("CINJURED", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("CINJSEV", "CRASH"), ("CINJURED", "CRASH"))
    G.add_edge(("ALCINV", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("DRGINV", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("CRASHYEAR", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("CRASHMONTH", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("DAYOFWEEK", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("CRASHTIME", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("EVENTS", "CRASH"), ("CASEID", "CRASH"))
    G.add_edge(("VEHICLES", "CRASH"), ("SUMMARY", "CRASH"))
    G.add_edge(("MANCOLL", "CRASH"), ("SUMMARY", "CRASH"))
    G.add_edge(("SUMMARY", "CRASH"), ("CASEID", "CRASH"))


    Veh_GV_node = ("VEHNO", "GV")
    G.add_edge(("VEHNO", "GV"), ("CASEID", "GV"))
    
    for var in [ "DVTOTAL", "CRASHTYPE", "PREFHE", "MODELYR", 
                "BODYTYPE", "BODYCAT", "SPEEDLIMIT", "CRITEVENT", "CRASHCONF","ALCTESTSRC","DRUGTEST","DAMPLANE"]:
        G.add_edge((var, "GV"), Veh_GV_node)
    G.add_edge(("DVLONG","GV"),("DVTOTAL","GV"))
    G.add_edge(("DVLAT","GV"),("DVTOTAL","GV"))
    G.add_edge(("ALCTESTRESULT", "GV"), ("ALCTESTSRC","GV"))
    G.add_edge(("ALCINV", "CRASH"), ("ALCTESTRESULT", "GV"), weight=p)
    G.add_edge(("DRGINV", "CRASH"), ("DRUGTEST", "GV"), weight=p)


    Veh_VPIC_node = ("VEHNO", "VPICDECODE")
    safety_node = ("ActiveSafetySysNote", "VPICDECODE")
    G.add_edge( Veh_VPIC_node,("CASEID", "VPICDECODE"))
    G.add_edge(("Model", "VPICDECODE"), Veh_VPIC_node)
    G.add_edge(("VehicleType", "VPICDECODE"), Veh_VPIC_node)
    G.add_edge(("ModelYear", "VPICDECODE"), Veh_VPIC_node)
    G.add_edge(safety_node, Veh_VPIC_node)
    G.add_edge(("AntilockBrakeSystem", "VPICDECODE"), safety_node)
    G.add_edge(("AdaptiveCruiseControl", "VPICDECODE"), safety_node)
    G.add_edge(("AutoPedestrianAlertingSound", "VPICDECODE"), safety_node)
    G.add_edge(("DynamicBrakeSupport", "VPICDECODE"), safety_node)
    G.add_edge(("ForwardCollisionWarning", "VPICDECODE"), safety_node)
    G.add_edge(("LaneCenteringAssistance", "VPICDECODE"), safety_node)
    G.add_edge(("LaneDepartureWarning", "VPICDECODE"), safety_node)
    G.add_edge(("LaneKeepingAssistance", "VPICDECODE"), safety_node)
    G.add_edge(("RearAutomaticEmergencyBraking", "VPICDECODE"), safety_node)
    # EVENTNO → CASEID
    EVENT_node = ("EVENTNO", "EVENT")
    G.add_edge(("EVENTNO", "EVENT"), ("CASEID", "EVENT"))
    for var in ["VEHNUM", "CLASS1", "GAD1", "CLASS2", "GAD2"]:
        G.add_edge((var, "EVENT"), EVENT_node)
        G.add_edge((var, "EVENT"), ("SUMMARY","CRASH"), weight=p)

    # OCCUPANT → VEHNO
    OCC_node = ("OCCNO", "OCC")
    role_node = ("ROLE", "OCC")
    G.add_edge(role_node, OCC_node)
    G.add_edge(("VEHNO", "OCC"), ("CASEID", "OCC"))
    G.add_edge( OCC_node, ("VEHNO", "OCC"))
    # for var in ["AGE", "HEIGHT", "WEIGHT", "ROLE", "SEATLOC", "SEX", "FETALMORT",
    #             "EYEWEAR", "POSTURE", "BELTUSE"]:
    for var in ["AGE", "HEIGHT", "SEATLOC", "FETALMORT","SEX","FETALMORT",
                "EYEWEAR", "POSTURE", "BELTUSE", "WEIGHT"]:
        G.add_edge((var, "OCC"), role_node)


    G.add_edge(("VEHNO", "AVOID"), ("CASEID", "AVOID"))
    G.add_edge(("EQUIP", "AVOID"), ("VEHNO", "AVOID"))
    G.add_edge( safety_node, ("ACTIVATE", "AVOID"),weight=p)
    G.add_edge(("AVAIL", "AVOID"), ("EQUIP", "AVOID"))
    G.add_edge(("ACTIVATE", "AVOID"), ("AVAIL", "AVOID"))
    
    G.add_edge(("VEHNO", "OCCONTACT"), ("CASEID", "OCCONTACT"))
    G.add_edge(("OCCNO", "OCCONTACT"), ("VEHNO", "OCCONTACT"))
    G.add_edge(("CONTCOMP", "OCCONTACT"), ("OCCNO", "OCCONTACT"))
    G.add_edge(("CONTAREA", "OCCONTACT"), ("CONTCOMP", "OCCONTACT"))
    # G.add_edge(("CONTACT", "OCCONTACT"), ("OCCNO", "OCCONTACT"))
    G.add_edge(("BODYREGION", "OCCONTACT"), ("CONTCOMP", "OCCONTACT"))
    G.add_edge(("BODYREGION", "OCCONTACT"), ("CONTAREA", "OCCONTACT"))
    G.add_edge(("EVIDENCE", "OCCONTACT"), ("BODYREGION", "OCCONTACT"))
    G.add_edge(("CONFIDENCE", "OCCONTACT"), ("EVIDENCE", "OCCONTACT"))
    
    
    
    return G


def find_all_paths_to_caseid(start_variable, start_sheet):
    G = build_variable_dependency_graph()
    start_node = (start_variable, start_sheet)
    target_node = ("CASEID", "center")
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
            if filtered_path and filtered_path[-1] == ("CASEID", "center"):
                
                filtered_path = filtered_path[:-1]

            if len(filtered_path) > 1:
                filtered_paths.append(filtered_path)

        return filtered_paths

    except nx.NetworkXNoPath:
        print("No path found.")
        return None



import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch




