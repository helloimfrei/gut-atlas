import re

def clean_feature_names(name):
    name = str(name)                          
    name = name.replace('.', '_')            
    name = re.sub(r'[^0-9a-zA-Z_]', '_', name)  
    return name

def dead_features(df):
    """
    Returns a list of features that have a total abundance of 0 across all samples

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with feature names as columns 

    Returns
    -------
    list
        A list of feature names that have a total abundance of 0
    """
    
    col_abundances = df.sum(numeric_only=True,axis = 0).sort_values().reset_index(name = 'total_abundance')
    drop_cols = col_abundances[col_abundances.total_abundance == 0]['index'].tolist()
    return drop_cols
