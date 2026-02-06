import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/data")
OUT_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/results")
OUT_DIR.mkdir(exist_ok=True)

CHUNKSIZE = 100_000

DEMO_COLS = ["CASEID", "PRIMARYID", "AGE", "AGE_COD", "SEX"]
DRUG_COLS = ["CASEID", "PRIMARYID", "DRUGNAME", "ROLE_COD"]
REAC_COLS = ["CASEID", "PRIMARYID", "PT"]

# FIXED: More flexible pattern matching
quarters = [
    p for p in BASE_DIR.iterdir()
    if p.is_dir()
]

# DEBUG: Print what directories were found
print(f"Found {len(quarters)} directories in {BASE_DIR}")
for q in quarters:
    print(f"  - {q.name}")

if not quarters:
    print("ERROR: No quarter directories found!")
    print(f"Contents of {BASE_DIR}:")
    for item in BASE_DIR.iterdir():
        print(f"  {item}")
    exit(1)

def find_faers_data_dir(quarter_dir: Path) -> Path:
    """
    Find the subdirectory containing FAERS .txt files
    """
    # First check if .txt files are directly in quarter_dir
    if list(quarter_dir.glob("*.txt")):
        print(f"  Found .txt files directly in {quarter_dir.name}")
        return quarter_dir
    
    # Otherwise search subdirectories
    for sub in quarter_dir.iterdir():
        if sub.is_dir():
            txt_files = list(sub.glob("*.txt"))
            if txt_files:
                print(f"  Found .txt files in {quarter_dir.name}/{sub.name}")
                return sub
    
    raise FileNotFoundError(f"No .txt files found in {quarter_dir}")

def get_available_columns(file_path: Path):
    """
    Read only the header to discover available columns.
    """
    try:
        return pd.read_csv(
            file_path,
            sep="$",
            encoding="latin-1",
            nrows=0
        ).columns.tolist()
    except Exception as e:
        print(f"  WARNING: Could not read header from {file_path.name}: {e}")
        return []

def load_table_from_quarter_chunked(
    quarter_dir: Path,
    table_name: str,
    desired_cols: list
):
    """
    Load table from quarter directory in chunks
    """
    try:
        data_dir = find_faers_data_dir(quarter_dir)
    except FileNotFoundError as e:
        print(f"  SKIP {quarter_dir.name}: {e}")
        return
    
    # More flexible file matching (case-insensitive)
    files = list(data_dir.glob(f"{table_name}*.txt")) + \
            list(data_dir.glob(f"{table_name.lower()}*.txt")) + \
            list(data_dir.glob(f"{table_name.upper()}*.txt"))
    
    if not files:
        print(f"  SKIP {quarter_dir.name}: No {table_name} files found")
        return

    print(f"  Processing {quarter_dir.name}/{table_name} ({len(files)} file(s))")
    
    for f in files:
        print(f"    Reading {f.name}...")
        available_cols = get_available_columns(f)
        
        if not available_cols:
            print(f"    SKIP {f.name}: Could not read columns")
            continue
        
        # Case-insensitive column matching
        available_cols_upper = [c.upper() for c in available_cols]
        usecols = []
        for desired_col in desired_cols:
            if desired_col.upper() in available_cols_upper:
                # Find the actual column name (might be different case)
                idx = available_cols_upper.index(desired_col.upper())
                usecols.append(available_cols[idx])
        
        if not usecols:
            print(f"    SKIP {f.name}: None of desired columns found")
            print(f"      Desired: {desired_cols}")
            print(f"      Available: {available_cols[:10]}...")
            continue
        
        print(f"    Using columns: {usecols}")
        
        chunk_count = 0
        row_count = 0
        
        try:
            for chunk in pd.read_csv(
                f,
                sep="$",
                encoding="latin-1",
                usecols=usecols,
                low_memory=False,
                chunksize=CHUNKSIZE
            ):
                # Standardize column names to uppercase
                chunk.columns = chunk.columns.str.upper()
                
                chunk["quarter"] = quarter_dir.name
                
                if table_name.upper() == "DRUG":
                    if "ROLE_COD" in chunk.columns:
                        before = len(chunk)
                        chunk = chunk[chunk["ROLE_COD"] == "PS"]
                        print(f"      Filtered DRUG: {before} → {len(chunk)} rows (PS only)")
                    
                    if "DRUGNAME" in chunk.columns:
                        chunk["DRUGNAME"] = chunk["DRUGNAME"].str.upper().str.strip()
                
                if not chunk.empty:
                    chunk_count += 1
                    row_count += len(chunk)
                    yield chunk
        
        except Exception as e:
            print(f"    ERROR reading {f.name}: {e}")
            continue
        
        print(f"    Processed {chunk_count} chunks, {row_count} rows from {f.name}")

def write_table_chunked(table_name: str, desired_cols: list):
    """
    Write table to parquet in chunks
    """
    out_path = OUT_DIR / f"{table_name.lower()}.parquet"
    writer = None
    total_rows = 0
    
    print(f"\n{'='*60}")
    print(f"Processing {table_name}")
    print(f"{'='*60}")
    
    for q in quarters:
        for chunk in load_table_from_quarter_chunked(q, table_name, desired_cols):
            if chunk.empty:
                continue
            
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
                print(f"\nCreated writer for {out_path}")
            
            writer.write_table(table)
            total_rows += len(chunk)
    
    if writer:
        writer.close()
        print(f"\n✓ Wrote {total_rows:,} rows to {out_path}")
        print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"\n✗ No data written for {table_name}")

if __name__ == "__main__":
    print("Starting FAERS ingestion (chunked, schema-safe)...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print()
    
    write_table_chunked("DEMO", DEMO_COLS)
    write_table_chunked("DRUG", DRUG_COLS)
    write_table_chunked("REAC", REAC_COLS)
    
    print("\n" + "="*60)
    print("FAERS ingestion completed")
    print("="*60)
    
    # Show what was created
    print("\nOutput files:")
    for f in OUT_DIR.glob("*.parquet"):
        print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")