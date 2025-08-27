import polars as pl

taxon_path = "../data/raw/taxonomic_table.csv"
metadata_path = "../data/raw/sample_metadata.tsv"
tags_path = "../data/raw/tags.tsv"

sample_metadata = pl.read_csv(metadata_path, separator="\t").with_columns(
    (pl.col("project") + "_" + pl.col("srr")).alias("sample")
)
sample_tags = pl.read_csv(tags_path, separator="\t").with_columns(
    (pl.col("project") + "_" + pl.col("srr")).alias("sample")
)

#taxon table is huge. open batches of 1000, normalize the abundances of each taxon on a per-sample basis
#then save each batch for easier processing going forward
reader = pl.read_csv_batched(taxon_path, batch_size=1000)
first_batch = True
i = 0
while True:
    batches = reader.next_batches(1)
    if not batches:
        break
    batch = batches[0]
    taxon_cols = batch.columns[2:]

    batch = batch.with_columns(
        pl.sum_horizontal([pl.col(col) for col in taxon_cols]).alias("total_reads")
    ).with_columns([
        (pl.col(col) / pl.col("total_reads")).alias(col) for col in taxon_cols
    ]).select(["sample"] + taxon_cols)

    merged = (
        batch.join(sample_metadata, on="sample", how="inner")
             .join(sample_tags, on="sample", how="left")
             .drop(["project_right", "srr_right", "srs_right", "total_bases", 
                    "instrument",'srs', 'project', 'srr', 'library_strategy', 'library_source'])
    )

    merged.write_parquet(f"../data/interim/processed_batches/taxa_merged_batch_{i}.parquet")
    i += 1


#lazily read all batches, split by region, and save each region separately
batches = pl.scan_parquet('../data/interim/processed_batches/taxa_merged_batch_*.parquet')

unique_regions = (
    batches.select("iso")
           .unique()
           .collect()
)

for region in unique_regions['iso']:
    split = batches.filter(pl.col('iso') == region).collect()

    split.write_parquet(f'../data/interim/regional_data/{region}_microbiome.parquet')