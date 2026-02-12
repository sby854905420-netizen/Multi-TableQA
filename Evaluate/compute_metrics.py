


def compute_metrics_BIRD(preds:list,labels:list) -> tuple[float,float]:
    exact_match_flag = []
    contain_flag = []
    for pred_row, label_row in zip(preds,labels):
        if pred_row == False:
            exact_match_flag.append(False)
            contain_flag.append(False)
        else:
            pred_values = set([list(d.values())[0] for d in pred_row])
            label_values = set([list(d.values())[0] for d in label_row])
            
            if pred_values == label_values:
                exact_match_flag.append(True)
                contain_flag.append(True)
            else:
                exact_match_flag.append(False)
                if pred_values > label_values:
                    contain_flag.append(True)
                else:
                    contain_flag.append(False)

    exact_accuray = round(sum(exact_match_flag) / len(exact_match_flag), 2)
    contain_accuray = round(sum(contain_flag) / len(contain_flag), 2)
    return exact_accuray,contain_accuray