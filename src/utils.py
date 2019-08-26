from texttable import Texttable

def tab_printer(args):
        
    args = vars(args)
    
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def create_numeric_mapping(node_properties):
    
    return {value:i for i, value in enumerate(node_properties)}
