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

quarters = [
    p for p in BASE_DIR.iterdir()
    if p.is_dir()
]
for q in quarters:
    print(f"  - {q.name}")

if not quarters:
    for item in BASE_DIR.iterdir():
        print(f"  {item}")
    exit(1)

def find_faers_data_dir(quarter_dir: Path) -> Path:
    if list(quarter_dir.glob("*.txt")):
        print(f"  Found .txt files directly in {quarter_dir.name}")
        return quarter_dir
    
   
    for sub in quarter_dir.iterdir():
        if sub.is_dir():
            txt_files = list(sub.glob("*.txt"))
            if txt_files:
                print(f"  Found .txt files in {quarter_dir.name}/{sub.name}")
                return sub
    
    raise FileNotFoundError(f"No .txt files found in {quarter_dir}")

def get_available_columns(file_path: Path):
    
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
  
    try:
        data_dir = find_faers_data_dir(quarter_dir)
    except FileNotFoundError as e:
        print(f"  SKIP {quarter_dir.name}: {e}")
        return
    
  
    files = list(data_dir.glob(f"{table_name}*.txt")) + \
            list(data_dir.glob(f"{table_name.lower()}*.txt")) + \
            list(data_dir.glob(f"{table_name.upper()}*.txt"))
    
    if not files:
        print(f"  SKIP {quarter_dir.name}: No {table_name} files found")
        return

    for f in files:

        available_cols = get_available_columns(f)
        
        if not available_cols:

            continue
        
        
        available_cols_upper = [c.upper() for c in available_cols]
        usecols = []
        for desired_col in desired_cols:
            if desired_col.upper() in available_cols_upper:
               
                idx = available_cols_upper.index(desired_col.upper())
                usecols.append(available_cols[idx])
        
        if not usecols:
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
        

def write_table_chunked(table_name: str, desired_cols: list):
 
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
            
            writer.write_table(table)
            total_rows += len(chunk)
    
    if writer:
        writer.close()

if __name__ == "__main__":
    write_table_chunked("DEMO", DEMO_COLS)
    write_table_chunked("DRUG", DRUG_COLS)
    write_table_chunked("REAC", REAC_COLS)
    
 