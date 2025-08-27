import re

def clean_feature_names(taxon:str):
    name = '_'.join(taxon.split('.'))
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)