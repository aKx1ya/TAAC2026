"""Generate schema.json from a Parquet file for HyFormer v0.2.

Enhanced version — additionally scans for ``item_dense_feats_*`` columns
so that item-side numerical attributes (price, rating, CTR history, etc.)
are included in the model's item_dense token.

Usage:
    python generate_schema.py <parquet_path> [output_json_path]
"""

import sys
import pyarrow.parquet as pq
import json


def generate_local_schema(parquet_path, output_json_path):
    print(f"Reading {parquet_path} ...")
    table = pq.read_table(parquet_path)

    schema_dict = {
        "user_int": [],
        "item_int": [],
        "user_dense": [],
        "item_dense": [],          # ← NEW: item-side numerical features
        "seq": {
            "seq_a": {"prefix": "seq_a", "ts_fid": None, "features": []},
            "seq_b": {"prefix": "seq_b", "ts_fid": None, "features": []},
            "seq_c": {"prefix": "seq_c", "ts_fid": None, "features": []},
            "seq_d": {"prefix": "seq_d", "ts_fid": None, "features": []},
        }
    }

    def get_stats(col_data):
        py_list = col_data.to_pylist()
        if not py_list:
            return 0, 1

        first_valid = next((x for x in py_list if x is not None), None)

        if isinstance(first_valid, list):
            flat = [val for sublist in py_list if sublist is not None
                    for val in sublist if val is not None]
            if not flat:
                return 0, 1
            max_val = max(flat)
            max_dim = max(len(sublist) if sublist is not None else 0
                         for sublist in py_list)
            return int(max_val) + 1, max(1, int(max_dim))
        else:
            valid_vals = [val for val in py_list if val is not None]
            if not valid_vals:
                return 0, 1
            return int(max(valid_vals)) + 1, 1

    print("Scanning feature columns ...")
    for col in table.column_names:
        # --- User Int ---
        if col.startswith('user_int_feats_'):
            fid = int(col.split('_')[-1])
            vs, dim = get_stats(table[col])
            schema_dict["user_int"].append([fid, vs, dim])

        # --- Item Int ---
        elif col.startswith('item_int_feats_'):
            fid = int(col.split('_')[-1])
            vs, dim = get_stats(table[col])
            schema_dict["item_int"].append([fid, vs, dim])

        # --- User Dense ---
        elif col.startswith('user_dense_feats_'):
            fid = int(col.split('_')[-1])
            _, dim = get_stats(table[col])
            schema_dict["user_dense"].append([fid, dim])

        # --- Item Dense (NEW) ---
        elif col.startswith('item_dense_feats_'):
            fid = int(col.split('_')[-1])
            _, dim = get_stats(table[col])
            schema_dict["item_dense"].append([fid, dim])

        # --- Sequence domains ---
        elif col.startswith('seq_'):
            parts = col.split('_')
            if len(parts) >= 3:
                domain = f"{parts[0]}_{parts[1]}"
                if domain in schema_dict["seq"]:
                    fid = int(parts[2])
                    vs, _ = get_stats(table[col])
                    schema_dict["seq"][domain]["features"].append([fid, vs])

    # Sort for deterministic output
    schema_dict["user_int"].sort(key=lambda x: x[0])
    schema_dict["item_int"].sort(key=lambda x: x[0])
    schema_dict["user_dense"].sort(key=lambda x: x[0])
    schema_dict["item_dense"].sort(key=lambda x: x[0])

    # Auto-detect timestamp column per sequence domain (last fid)
    for domain in list(schema_dict["seq"].keys()):
        features = schema_dict["seq"][domain]["features"]
        if len(features) == 0:
            del schema_dict["seq"][domain]
            continue
        features.sort(key=lambda x: x[0])
        fids = [x[0] for x in features]
        schema_dict["seq"][domain]["ts_fid"] = fids[-1] if len(fids) > 0 else None

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(schema_dict, f, indent=2)
    print(f"Schema written to {output_json_path}")
    print(f"  user_int:    {len(schema_dict['user_int'])} features")
    print(f"  item_int:    {len(schema_dict['item_int'])} features")
    print(f"  user_dense:  {len(schema_dict['user_dense'])} features")
    print(f"  item_dense:  {len(schema_dict['item_dense'])} features")
    print(f"  seq domains: {list(schema_dict['seq'].keys())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_schema.py <parquet_path> [output_json_path]")
        sys.exit(1)
    parquet_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else './data/schema.json'
    generate_local_schema(parquet_path, output_path)
