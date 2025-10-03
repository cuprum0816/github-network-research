#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大图安全版：构建贡献者-贡献者投影网络
支持任意仓库贡献者数量（分块写入 CSV 防止内存爆掉）
输出：results/contributors_edgelist.csv 和 contributors_graph.gexf
"""

import os
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import igraph as ig
from colorama import init, Fore
from collections import defaultdict

init(autoreset=True)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE = "./repo_to_contribs_top1000.parquet"
MIN_CONTRIBS_PER_REPO = 2
EDGE_CSV = os.path.join(RESULTS_DIR, "contributors_edgelist.csv")
CHUNK_SIZE = 1_000_000  # 每100万条边写一次

# ------------------ 读取数据 ------------------
def read_data():
    print(Fore.BLUE + f"读取数据: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    if 'Repo_ID' not in df.columns or 'Contributors' not in df.columns:
        raise ValueError("文件必须包含 'Repo_ID' 和 'Contributors' 列")
    print(Fore.GREEN + f"数据读取完成，共 {len(df)} 行")
    return df

# ------------------ 构建边 CSV（分块） ------------------
def build_edge_csv(df):
    edge_list = []
    print(Fore.BLUE + "构建边列表（分块写入 CSV）...")
    if os.path.exists(EDGE_CSV):
        os.remove(EDGE_CSV)

    node_set = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="仓库处理"):
        contribs = row['Contributors']
        if len(contribs) < MIN_CONTRIBS_PER_REPO:
            continue
        node_set.update(contribs)
        for a, b in combinations(contribs, 2):
            if a < b:
                edge_list.append((a, b, 1))
            else:
                edge_list.append((b, a, 1))
        if len(edge_list) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(edge_list, columns=['source','target','weight'])
            df_chunk.to_csv(EDGE_CSV, mode='a', index=False, header=not os.path.exists(EDGE_CSV))
            edge_list = []

    # 最后剩余的写入
    if edge_list:
        df_chunk = pd.DataFrame(edge_list, columns=['source','target','weight'])
        df_chunk.to_csv(EDGE_CSV, mode='a', index=False, header=not os.path.exists(EDGE_CSV))

    print(Fore.GREEN + f"边列表 CSV 保存完成: {EDGE_CSV}")
    return node_set

# ------------------ 构建 igraph 图 ------------------
def build_graph_from_csv():
    print(Fore.BLUE + "从 CSV 构建 igraph 图...")
    df_edges = pd.read_csv(EDGE_CSV)
    # 聚合边权
    df_agg = df_edges.groupby(['source','target'], as_index=False)['weight'].sum()
    G = ig.Graph.TupleList(df_agg.itertuples(index=False), weights=True, directed=False)
    print(Fore.GREEN + f"igraph 图构建完成: 节点数 {G.vcount()}, 边数 {G.ecount()}")
    return G

# ------------------ 保存图 ------------------
def save_graph(G):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    graphml_path = os.path.join(RESULTS_DIR, "contributors_graph.graphml")
    G.write_graphml(graphml_path)
    print(Fore.GREEN + f"GraphML 保存完成: {graphml_path}")
    pickle_path = os.path.join(RESULTS_DIR, "contributors_graph.pickle")
    G.write_pickle(pickle_path)
    print(Fore.GREEN + f"Pickle 保存完成: {pickle_path}")
    try:
        import networkx as nx
        from igraph import Graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(G.vs['name'])
        nx_graph.add_weighted_edges_from([
            (G.vs[e.source]['name'], G.vs[e.target]['name'], e['weight'])
            for e in G.es
        ])
        gexf_path = os.path.join(RESULTS_DIR, "contributors_graph.gexf")
        nx.write_gexf(nx_graph, gexf_path)
        print(Fore.GREEN + f"GEXF 保存完成: {gexf_path}")
    except ImportError:
        print(Fore.YELLOW + "未安装 networkx，跳过 GEXF 导出。")

def main():
    df = read_data()
    node_set = build_edge_csv(df)
    G = build_graph_from_csv()
    save_graph(G)
    print(Fore.GREEN + "图构建完成 ✅")

if __name__ == "__main__":
    main()
