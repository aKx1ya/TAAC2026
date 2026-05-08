import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import os

# ==========================================
# 1. 基础配置与数据加载
# ==========================================
# 读取本地 parquet 文件
# 1. 自动获取当前 python 脚本 (data_analysis.py) 所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出 parquet 文件的绝对路径 (假设 parquet 文件也放在 data 文件夹下)
file_path = os.path.join(current_dir, 'demo_1000.parquet')
try:
    df = pd.read_parquet(file_path)
    print(f"✅ 成功加载数据，形状: {df.shape}")
except Exception as e:
    print(f"❌ 读取失败，请检查依赖 (pip install pyarrow) 或文件路径: {e}")
    exit()

# 根据 TAAC2026 Schema 定义特征组
user_list_feats = [f'user_int_feats_{i}' for i in [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]]
domain_seq_feats = {
    'Domain_A': [f'domain_a_seq_{i}' for i in range(38, 47)],
    'Domain_B': [f'domain_b_seq_{i}' for i in range(67, 80)] + ['domain_b_seq_88'],
    'Domain_C': [f'domain_c_seq_{i}' for i in range(27, 38)] + ['domain_c_seq_47'],
    'Domain_D': [f'domain_d_seq_{i}' for i in range(17, 27)]
}

# 扁平化所有的 sequence 列表，方便后续循环
all_seq_feats = list(itertools.chain.from_iterable(domain_seq_feats.values()))

# ==========================================
# 2. 核心验证：ID 空间重合度扫描 (Jaccard Similarity)
# ==========================================
def calc_jaccard(list1, list2):
    """计算两个列表的 Jaccard 相似度"""
    # 处理 None、NaN 或非列表类型
    if not isinstance(list1, (list, np.ndarray)) or not isinstance(list2, (list, np.ndarray)):
        return 0.0
    s1, s2 = set(list1), set(list2)
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

print("\n🔍 正在执行全量多值特征与序列特征的重合度扫描...")
overlap_results = []

for u_feat in user_list_feats:
    if u_feat not in df.columns: continue
    
    for seq_feat in all_seq_feats:
        if seq_feat not in df.columns: continue
        
        # 逐行计算 Jaccard 相似度
        jaccard_scores = df.apply(lambda row: calc_jaccard(row[u_feat], row[seq_feat]), axis=1)
        
        # 统计非零重叠的样本比例 和 平均相似度
        non_zero_ratio = (jaccard_scores > 0).mean()
        mean_sim = jaccard_scores.mean()
        
        if non_zero_ratio > 0.01: # 仅记录有哪怕一丁点重叠的组合
            overlap_results.append({
                'User_Feature': u_feat,
                'Sequence_Feature': seq_feat,
                'Non_Zero_Overlap_Ratio': non_zero_ratio,
                'Mean_Jaccard': mean_sim
            })

# 输出高潜力的特征对
res_df = pd.DataFrame(overlap_results)
if not res_df.empty:
    res_df = res_df.sort_values(by='Mean_Jaccard', ascending=False)
    print("\n🔥 Top 10 ID 空间重合度最高的特征对 (建议做强交叉/Attention):")
    print(res_df.head(10).to_string(index=False))
else:
    print("\n⚠️ 未发现显著的 ID 空间重合。各 Domain 序列与 User 多值特征可能处于独立编码空间。")

# ==========================================
# 3. 可视化分析：特征分布与关联
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 图 1：如果存在重叠，画出 Top 组合的热力图矩阵
if not res_df.empty:
    top_u_feats = res_df['User_Feature'].unique()[:5]
    top_s_feats = res_df['Sequence_Feature'].unique()[:5]
    
    pivot_df = pd.DataFrame(index=top_u_feats, columns=top_s_feats, data=0.0)
    for _, row in res_df.iterrows():
        if row['User_Feature'] in top_u_feats and row['Sequence_Feature'] in top_s_feats:
            pivot_df.loc[row['User_Feature'], row['Sequence_Feature']] = row['Mean_Jaccard']
            
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("User多值特征与Sequence特征的平均 Jaccard 相似度")
    plt.tight_layout()
    plt.savefig('top_features_heatmap.png')
    plt.show()

# 图 2：序列长度与 User 单值画像的关联
# 选取一个典型的标量特征 (例如 user_int_feats_1) 和 某个 Domain 的长度
scalar_feat = 'user_int_feats_1'
test_seq = 'domain_a_seq_38' 

if scalar_feat in df.columns and test_seq in df.columns:
    df['seq_len'] = df[test_seq].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=scalar_feat, y='seq_len', data=df, palette="Set2")
    plt.title(f"User单值特征 ({scalar_feat}) 对 {test_seq} 序列长度的影响")
    plt.xlabel(scalar_feat)
    plt.ylabel("Sequence Length")
    plt.tight_layout()
    plt.savefig('scalar_vs_seqlen.png')
    plt.show()

# 图 3：User 多值特征长度 vs Domain 序列长度的散点关系
test_u_list = 'user_int_feats_15'
if test_u_list in df.columns and test_seq in df.columns:
    df['u_list_len'] = df[test_u_list].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='u_list_len', y='seq_len', data=df, alpha=0.6, color='coral')
    plt.title(f"User特征长度 ({test_u_list}) vs Sequence长度 ({test_seq})")
    plt.xlabel(f"Length of {test_u_list}")
    plt.ylabel(f"Length of {test_seq}")
    # 添加拟合线
    sns.regplot(x='u_list_len', y='seq_len', data=df, scatter=False, color='darkred', line_kws={"linewidth": 1})
    plt.tight_layout()
    plt.savefig('length_correlation.png')
    plt.show()

print("\n✅ 分析完成！所有可视化图表已保存至当前目录并展示。")