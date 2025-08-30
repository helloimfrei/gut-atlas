import re

def clean_feature_names(name):
    name = str(name)                          
    name = name.replace('.', '_')            
    name = re.sub(r'[^0-9a-zA-Z_]', '_', name)  
    return name