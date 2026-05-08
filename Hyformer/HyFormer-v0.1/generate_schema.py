import pyarrow.parquet as pq
import json

def generate_local_schema(parquet_path, output_json_path):
    print(f"正在读取 {parquet_path} ...")
    table = pq.read_table(parquet_path)
    
    schema_dict = {
        "user_int": [],
        "item_int": [],
        "user_dense": [],
        "seq": {
            "seq_a": {"prefix": "seq_a", "ts_fid": None, "features": []},
            "seq_b": {"prefix": "seq_b", "ts_fid": None, "features": []},
            "seq_c": {"prefix": "seq_c", "ts_fid": None, "features": []},
            "seq_d": {"prefix": "seq_d", "ts_fid": None, "features": []}
        }
    }
    
    # 【更新点】使用纯 Python 列表解析，彻底避开 pyarrow.compute 的类型转换 bug
    def get_stats(col_data):
        py_list = col_data.to_pylist()
        if not py_list:
            return 0, 1
            
        # 判断第一行数据是不是列表（变长特征）
        # 需要跳过可能是 None 的空值
        first_valid = next((x for x in py_list if x is not None), None)
        
        if isinstance(first_valid, list):
            # 展平所有列表，并过滤掉空值
            flat = [val for sublist in py_list if sublist is not None for val in sublist if val is not None]
            if not flat:
                return 0, 1
            max_val = max(flat)
            max_dim = max(len(sublist) if sublist is not None else 0 for sublist in py_list)
            return int(max_val) + 1, max(1, int(max_dim))
        else:
            # 标量特征
            valid_vals = [val for val in py_list if val is not None]
            if not valid_vals:
                return 0, 1
            return int(max(valid_vals)) + 1, 1

    print("正在扫描特征维度与词表大小...")
    for col in table.column_names:
        # 解析 User Int
        if col.startswith('user_int_feats_'):
            fid = int(col.split('_')[-1])
            vs, dim = get_stats(table[col])
            schema_dict["user_int"].append([fid, vs, dim])
            
        # 解析 Item Int
        elif col.startswith('item_int_feats_'):
            fid = int(col.split('_')[-1])
            vs, dim = get_stats(table[col])
            schema_dict["item_int"].append([fid, vs, dim])
            
        # 解析 User Dense
        elif col.startswith('user_dense_feats_'):
            fid = int(col.split('_')[-1])
            _, dim = get_stats(table[col])
            schema_dict["user_dense"].append([fid, dim])
            
        # 解析 Sequence 域
        elif col.startswith('seq_'):
            parts = col.split('_')
            if len(parts) >= 3:
                domain = f"{parts[0]}_{parts[1]}"  # 例如 seq_a
                if domain in schema_dict["seq"]:
                    fid = int(parts[2])
                    vs, _ = get_stats(table[col])
                    schema_dict["seq"][domain]["features"].append([fid, vs])

    # 排序以保证模型对齐稳定
    schema_dict["user_int"].sort(key=lambda x: x[0])
    schema_dict["item_int"].sort(key=lambda x: x[0])
    schema_dict["user_dense"].sort(key=lambda x: x[0])
    
    # 自动探测序列的时间戳列
    for domain in list(schema_dict["seq"].keys()):
        features = schema_dict["seq"][domain]["features"]
        if len(features) == 0:
            del schema_dict["seq"][domain]  # 删除不存在的空域
            continue
        features.sort(key=lambda x: x[0])
        fids = [x[0] for x in features]
        schema_dict["seq"][domain]["ts_fid"] = fids[-1] if len(fids) > 0 else None

    # 保存 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(schema_dict, f, indent=2)
    print(f"成功！已基于实际数据生成专属 Schema：{output_json_path}")

if __name__ == "__main__":
    generate_local_schema('./data/demo_1000.parquet', './data/schema.json')