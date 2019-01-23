import os
from torch import nn

def get_expt_name(config_file, configs_dict):
    path, base_name = os.path.split(config_file)
    base_name = os.path.splitext(base_name)[0]
    holding_dir = os.path.split(path)[1]
    return os.path.join(holding_dir, base_name)
 

def expt_summary(expt_name, configs):
    barrier_str = "#"*80
    return (barrier_str + "\n\n" + expt_name + "\n\n" + barrier_str
            + "\n\n" + configs.get("desc", "NO DESCRIPTION") + "\n\n")
    

def feed_forward_layer_list(input_size, params):
    last_size = input_size
    out = []
    for e in params:
        if isinstance(e, int):
            out.append(nn.Linear(last_size, e)) 
            last_size = e 
        elif isinstance(e, str):
           if e == "BatchNorm1d":
               out.append(nn.BatchNorm1d(last_size))
           else:
               out.append(getattr(nn, e)())
        else:
           raise NotImplementedError

    return out 

