import pandas as pd
import os

def filter_by_tag(batch_dir,tag_list):
    """
    Filter the tag column of microbiome dataset using given tag_list, return merged, filtered DataFrame 
    """
    merged = pd.DataFrame()
    batches = os.listdir(batch_dir)
    for idx,file in enumerate(batches):
        print('starting batch',idx+1,'of',len(batches))
        df = pd.read_parquet(os.path.join(batch_dir,file))
        df = df[df.tag.isin(tag_list)]
        merged = pd.concat([merged,df],ignore_index=True)
        del df
    print('finished',idx)
    return merged

# chat saved me some tedium here. cleaning up the values column and converting to binary diseased(1) or not diseased (0)
def map_gi_status_binary(value):
    """
    Map GI-related values (categorical or numeric IBS-SSS) to binary 0/1.
    
    Args:
        value (str or int): raw tag value from dataset
    
    Returns:
        int: 0 = healthy/none, 1 = any GI issue
    """
    if value is None:
        return 0
    
    val = str(value).strip().lower()
    
    # ---- Handle numeric IBS-SSS scores ----
    if val.isdigit():
        score = int(val)
        return 1 if score >= 75 else 0
    
    # ---- Known healthy/negative markers ----
    healthy_markers = {
        "i do not have this condition", "none", "normal", "healthy",
        "no", "no diagnosed disorders", "not applicable",
        "not_applicable", "na", "unk", "not provided", "unspecified",
        "not collected", "control", "labcontrol test", "hc", "healthy_control"
    }
    
    if val in healthy_markers:
        return 0
    
    # ---- Otherwise assume positive GI condition ----
    return 1
