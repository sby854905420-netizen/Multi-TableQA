import os
import sqlite3
import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
db_path = os.path.join(CURRENT_DIR, "SqlDatabase/financial.sqlite")
output_path = os.path.join(CURRENT_DIR, "DataBaseInfo/data_samples.xlsx")

conn = sqlite3.connect(db_path)
query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
tables = pd.read_sql_query(query, conn)['name'].tolist()

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for table_name in tables:
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 30;', conn)
            df.to_excel(writer, sheet_name=table_name[:31], index=False) 
            print(f"✅ {table_name}")
        except Exception as e:
            print(f"{table_name} — {e}")

conn.close()
print(f"✔️ {output_path}")
