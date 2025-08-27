import pandas as pd
from src.gutatlas.features import clean_feature_names
from src.gutatlas.data import map_gi_status_binary, filter_by_tag
from sort_tags import gi_tags


gi_merged = filter_by_tag('../data/interim/regional_data',gi_tags)
gi_merged.to_parquet('../data/interim/filtered_and_merged/gi_microbiomes_merged.parquet')

#remove duplicate rows for each sample. only need to know if any disease is present or not 
merged_gi = pd.read_parquet('../data/interim/filtered_and_merged/gi_microbiomes_merged.parquet')
merged_gi['disease_present'] = merged_gi.value.apply(map_gi_status_binary)

gi_training = (merged_gi
                         .sort_values(by = ['disease_present','sample'],ascending=False)
                         .drop_duplicates(subset = 'sample',keep='first')
                         .reset_index(drop=True)
                         .drop(columns = ['pubdate','geo_loc_name','iso','region','tag','value'])
                         .rename(columns = {col:clean_feature_names(col) for col in merged_gi.columns})
                         )


gi_training.to_parquet('../data/processed/gi_training.parquet')