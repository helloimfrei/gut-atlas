#!/usr/bin/env python3
"""
Build binary classification dataset for GI disease prediction.

This script processes raw microbiome data from Microbiomap to create a binary
classification dataset for predicting GI disease presence.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import skbio.stats.composition as composition

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gutatlas.data import filter_by_tag, map_gi_status_binary
from gutatlas.features import clean_feature_names, dead_features
from gutatlas.utils.constants import GI_TAGS


def process_taxonomic_batches(
    taxon_path: str,
    metadata_path: str,
    tags_path: str,
    batch_output_dir: str,
    batch_size: int = 1000
) -> None:
    """Process taxonomic data in batches and merge with metadata."""
    print(f"Loading metadata from {metadata_path}...")
    sample_metadata = pl.read_csv(metadata_path, separator="\t").with_columns(
        (pl.col("project") + "_" + pl.col("srr")).alias("sample")
    )

    print(f"Loading tags from {tags_path}...")
    sample_tags = pl.read_csv(tags_path, separator="\t").with_columns(
        (pl.col("project") + "_" + pl.col("srr")).alias("sample")
    )

    Path(batch_output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing taxonomic data in batches of {batch_size}...")
    reader = pl.read_csv_batched(taxon_path, batch_size=batch_size)
    batch_num = 0

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break

        batch = batches[0]
        taxon_cols = batch.columns[2:]

        # Total sum scaling (TSS) normalization
        batch = (
            batch.with_columns(
                pl.sum_horizontal([pl.col(col) for col in taxon_cols]).alias("total_reads")
            )
            .with_columns(
                [(pl.col(col) / pl.col("total_reads")).alias(col) for col in taxon_cols]
            )
            .select(["sample"] + taxon_cols)
        )

        # Merge with metadata and tags
        merged = (
            batch.join(sample_metadata, on="sample", how="inner")
            .join(sample_tags, on="sample", how="left")
            .drop([
                "project_right", "srr_right", "srs_right", "total_bases",
                "instrument", "srs", "project", "srr", "library_strategy",
                "library_source"
            ])
        )

        output_path = f"{batch_output_dir}/taxa_merged_batch_{batch_num}.parquet"
        merged.write_parquet(output_path)
        batch_num += 1

        if batch_num % 10 == 0:
            print(f"  Processed {batch_num} batches...")

    print(f"Completed processing {batch_num} batches")


def split_by_region(batch_dir: str, regional_output_dir: str) -> None:
    """Split batched data by geographic region."""
    print("Splitting data by region...")
    Path(regional_output_dir).mkdir(parents=True, exist_ok=True)

    batches = pl.scan_parquet(f"{batch_dir}/taxa_merged_batch_*.parquet")
    unique_regions = batches.select("iso").unique().collect()

    for region in unique_regions["iso"]:
        split = batches.filter(pl.col("iso") == region).collect()
        output_path = f"{regional_output_dir}/{region}_microbiome.parquet"
        split.write_parquet(output_path)
        print(f"  Saved {region}: {len(split)} samples")


def create_binary_classification_dataset(
    regional_data_dir: str,
    output_path: str
) -> None:
    """Create binary classification dataset for GI disease prediction."""
    print(f"Filtering data by GI tags ({len(GI_TAGS)} tags)...")
    gi_merged = filter_by_tag(regional_data_dir, GI_TAGS)

    print("Converting to pandas and processing...")
    merged_gi = gi_merged.to_pandas()

    print("Removing dead features...")
    drop_cols = dead_features(merged_gi)
    print(f"  Found {len(drop_cols)} dead features")

    print("Mapping disease status to binary...")
    merged_gi["disease_present"] = merged_gi.value.apply(map_gi_status_binary)

    print("Removing duplicate samples...")
    gi_training = (
        merged_gi.sort_values(by=["disease_present", "sample"], ascending=False)
        .drop_duplicates(subset="sample", keep="first")
        .reset_index(drop=True)
        .drop(columns=["pubdate", "geo_loc_name", "iso", "region", "tag", "value"] + drop_cols)
    )

    print("Cleaning feature names...")
    gi_training = gi_training.rename(
        columns={col: clean_feature_names(col) for col in gi_training.columns}
    )

    print("Removing shallow taxa (below family level)...")
    def drop_shallow_taxa(col: str) -> bool:
        levels = col.split("_")
        return "na" in levels[:5]

    keep_cols = [col for col in gi_training.columns if not drop_shallow_taxa(col)]
    gi_training_deepcols = gi_training[keep_cols]
    print(f"  Retained {len(keep_cols)} features")

    print("Applying centered log-ratio (CLR) transformation...")
    abundances = gi_training_deepcols.drop(columns=["sample", "disease_present"])

    zero_rows = (abundances == 0).all(axis=1)
    abundances_nonzero = abundances[~zero_rows]
    print(f"  Removed {zero_rows.sum()} samples with all-zero abundances")

    abundances_pseudo = composition.multi_replace(abundances_nonzero)
    abundance_clr = composition.clr(abundances_pseudo)

    gi_features_clr = pd.DataFrame(
        abundance_clr,
        columns=abundances_nonzero.columns,
        index=abundances_nonzero.index
    )

    gi_features_clr['sample'] = gi_training_deepcols.loc[gi_features_clr.index, 'sample']
    gi_features_clr['disease_present'] = gi_training_deepcols.loc[gi_features_clr.index, 'disease_present']
    gi_features_clr = gi_features_clr.reset_index(drop=True)

    gi_binclass_training_set = gi_features_clr.drop(columns=["sample"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gi_binclass_training_set.to_parquet(output_path)

    n_samples = len(gi_binclass_training_set)
    n_features = len(gi_binclass_training_set.columns) - 1
    n_diseased = gi_binclass_training_set['disease_present'].sum()
    n_healthy = n_samples - n_diseased

    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"Total samples:     {n_samples:,}")
    print(f"Healthy (y=0):     {n_healthy:,} ({n_healthy/n_samples*100:.1f}%)")
    print(f"Diseased (y=1):    {n_diseased:,} ({n_diseased/n_samples*100:.1f}%)")
    print(f"Features:          {n_features:,} bacterial taxa")
    print(f"Output saved to:   {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Build binary classification dataset for GI disease prediction"
    )

    parser.add_argument(
        "--taxon-path",
        type=str,
        required=True,
        help="Path to taxonomic table CSV file"
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Path to sample metadata TSV file"
    )
    parser.add_argument(
        "--tags-path",
        type=str,
        required=True,
        help="Path to tags TSV file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save final training dataset (parquet)"
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="./data/interim/batches",
        help="Directory for intermediate batch files"
    )
    parser.add_argument(
        "--regional-dir",
        type=str,
        default="./data/interim/regional_data",
        help="Directory for region-specific data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing taxonomic data"
    )
    parser.add_argument(
        "--skip-batching",
        action="store_true",
        help="Skip batch processing (use existing batches)"
    )
    parser.add_argument(
        "--skip-regional",
        action="store_true",
        help="Skip regional splitting (use existing regional data)"
    )

    args = parser.parse_args()

    if not args.skip_batching:
        process_taxonomic_batches(
            args.taxon_path,
            args.metadata_path,
            args.tags_path,
            args.batch_dir,
            args.batch_size
        )
    else:
        print("Skipping batch processing (using existing batches)")

    if not args.skip_regional:
        split_by_region(args.batch_dir, args.regional_dir)
    else:
        print("Skipping regional splitting (using existing regional data)")

    create_binary_classification_dataset(
        args.regional_dir,
        args.output_path
    )


if __name__ == "__main__":
    main()
