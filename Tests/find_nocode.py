import pandas as pd
from variable_Graph import find_all_paths_to_caseid
from collections.abc import Iterable

def flatten(iterable):
    for item in iterable:
        if isinstance(item, dict):
            for key in item.keys():
                yield key
            for value in item.values():
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                    yield from flatten(value)
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def is_complete_subpath(path1, path2):
   
    len1, len2 = len(path1), len(path2)
    if len1 > len2:
        return False
    for i in range(len2 - len1 + 1):
        if path2[i:i+len1] == path1:
            return True
    return False

def is_contiguous_subpath(path1, path2):
    return is_complete_subpath(path1, path2)

def search_on_graph(variables, sheet_names):
    all_variables = []
    all_sheets = []
    unique_paths = []

    for i in range(len(variables)):
        for path in find_all_paths_to_caseid(variables[i], sheet_names[i]) or []:
            keep_path = True
            paths_to_remove = []

            for existing_path in unique_paths:
                if is_contiguous_subpath(path, existing_path):
                   
                    keep_path = False
                    break
                elif is_contiguous_subpath(existing_path, path):
              
                    paths_to_remove.append(existing_path)


            for p in paths_to_remove:
                unique_paths.remove(p)
                idx = all_variables.index([var for var, sheet in p])
                del all_variables[idx]
                del all_sheets[idx]

            if keep_path:
                unique_paths.append(path)
                vars_this_path = [var for var, sheet in path]
                sheets_this_path = [sheet for var, sheet in path]
                all_variables.append(vars_this_path)
                all_sheets.append(sheets_this_path[0] if sheets_this_path else None)

    return all_variables, all_sheets




def create_table_description(all_variables):
    flattened = list(flatten(all_variables))
    unique_variables = set(flattened)
    indices = [v for v in unique_variables if v not in {"VEHNO", "OCCNO", "CASEID","EVENTNO"}]
    df = pd.read_excel("notebooks/table_description_wordcloud.xlsx")
    
    with open("data/logs/temporary_indices.txt", "w", encoding="utf-8") as file:    
        file.write("\n"+"**Relevant variables and descriptions:**\n" )
        # file.write("*Variable:*".join( f" {df.iloc[i]['name']}: {df.iloc[i]['description']} *Code mapping (value → meaning):*: {df.iloc[i]['format']}\n" for i in indices) + "\n")
        for name in indices:
            row = df[df['name'] == name].iloc[0]  
            # file.write(f"*Variable:* {row['name']}: *Code mapping (value → meaning):*: {row['format']}\n")
            file.write(f"*Variable:* {row['name']}: {row['description']}  *Code mapping (value → meaning):*: {row['format']}\n")

def create_variables_record(all_variables, all_sheets):
    
   with open("data/logs/indices_record.txt", "a", encoding="utf-8") as file:
        file.write("\n"+"**Relevant variables and descriptions:**\n" )
        for var_list, sheet_name in zip(all_variables, all_sheets):
            file.write(f"Sheet: {sheet_name}\n")
            for var in var_list:
                file.write(f"Variable: {var}\n")
            file.write("\n")
            
def find(case_id, variables, sheet_names):
    all_variables, all_sheets = search_on_graph( variables, sheet_names)
    flag = 0
    table_dp = pd.read_excel("notebooks/table_description_wordcloud_new.xlsx")
    for i in range(17, 24): 
        file_path = f'data/processed_data/case_info_20{i}.xlsx'

        for var_list, sheet_name in zip(all_variables, all_sheets):
            # print(var_list, sheet_name)
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception:
                continue

            if 'CASEID' not in df.columns:
                continue

            matched_rows = df[df['CASEID'] == case_id]
            if matched_rows.empty:
                continue

            result_lines = []

            for idx, row in matched_rows.iterrows():
                flag = 1
                vehno = row['VEHNO'] if 'VEHNO' in row else None
                occno = row['OCCNO'] if 'OCCNO' in row else None
                eventno = row['EVENTNO'] if 'EVENTNO' in row else None
                
                main_vars = [v for v in var_list if v not in {"VEHNO", "OCCNO", "CASEID","EVENTNO"}]

                if main_vars:
                    var_parts = []  

                    for var in reversed(main_vars):

                        var_info = table_dp[table_dp['name'] == var]


                        # if not var_info.empty:
                        # print(var_info)
                        var_format = var_info.iloc[0]['format']  
                        var_name = var_info.iloc[0]['fullname'] 
                        if var in df.columns:
                            value = row[var]

                            if pd.isna(value):
                                var_parts.append(f"the {var} is: Nan")
                            else:
                                if var_format == "nan":
                                    var_parts.append(f"the {var_name} is: {value}")
                                else:
                                
                                    if isinstance(var_format, str):
                                        var_format = eval(var_format)
                                    if isinstance(var_format, dict):
                    
                                        if value in var_format:
                                            var_parts.append(f"the {var_name} is: {var_format[value]}")
                                        else:
                                            var_parts.append(f"the {var_name} is: {value}")
                                    else:
                                 
                                        var_parts.append(f"the {var_name} is: {value}")

                    if var_parts:
                 
                        if vehno is not None and occno is not None:
                            line = f"In case {case_id}, for vehicle NO.{vehno}, occupant NO.{occno}, " + ", ".join(var_parts)
                        elif vehno is not None:
                            line = f"In case {case_id}, for vehicle NO.{vehno}, " + ", ".join(var_parts)
                        elif eventno is not None:
                            line = f"In case {case_id}, for event NO.{eventno}, " + ", ".join(var_parts)
                        else:
                            line = f"In case {case_id}, " + ", ".join(var_parts)

                        result_lines.append(line)
            
            if result_lines:
                with open("data/logs/output.txt", "a", encoding="utf-8") as file: 
                    for line in result_lines:
                        file.write(line + "\n")
                        print(line)
    if flag == 0:
        line = f"The case {case_id} is not in the datasets."
        with open("data/logs/output.txt", "a", encoding="utf-8") as file:
            file.write(line + "\n")
            print(line)
                
                        


