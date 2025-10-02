#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建贡献者-贡献者投影网络并计算网络指标（igraph版，适合大图）
输入文件：repo_to_contribs_top1000.parquet (列: Repo_ID, Contributors)
输出文件：./results/contributors_graph.gexf, contributors_edgelist.csv, contributors_metrics.parquet
"""

import os
from collections import defaultdict
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import igraph as ig
from colorama import init, Fore
import argparse
import random

init(autoreset=True)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE = "./repo_to_contribs_top1000.parquet"
MIN_CONTRIBS_PER_REPO = 2
DEFAULT_APPROX_K = 1000  # 默认采样源节点数

def read_data():
    print(Fore.BLUE + f"读取数据: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    if 'Repo_ID' not in df.columns or 'Contributors' not in df.columns:
        raise ValueError("文件必须包含 'Repo_ID' 和 'Contributors' 列")
    print(Fore.GREEN + f"数据读取完成，共 {len(df)} 行")
    return df

def build_contrib_edges(df, min_contribs_per_repo=2):
    edge_weights = defaultdict(int)
    node_set = set()
    print(Fore.BLUE + "构建贡献者-贡献者投影网络权重...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="仓库处理"):
        contribs = row['Contributors']
        if len(contribs) < min_contribs_per_repo:
            continue
        node_set.update(contribs)
        for a, b in combinations(contribs, 2):
            if a < b:
                edge_weights[(a, b)] += 1
            else:
                edge_weights[(b, a)] += 1
    print(Fore.GREEN + f"共 {len(node_set)} 个节点，{len(edge_weights)} 条边")
    return edge_weights, node_set

def build_igraph_graph(edge_weights, node_set):
    node_list = list(node_set)
    node_index = {n: i for i, n in enumerate(node_list)}
    edges = [(node_index[a], node_index[b]) for (a, b) in edge_weights.keys()]
    weights = list(edge_weights.values())
    G = ig.Graph(edges=edges, directed=False)
    G.vs['name'] = node_list
    G.es['weight'] = weights
    print(Fore.GREEN + f"igraph 图构建完成: 节点数 {G.vcount()}, 边数 {G.ecount()}")
    return G

def compute_metrics(G, approx_k=DEFAULT_APPROX_K):
    print(Fore.BLUE + "计算 Degree ...")
    degree = G.degree()
    print(Fore.BLUE + "计算 Closeness ...")
    closeness = G.closeness()
    print(Fore.BLUE + "计算 PageRank ...")
    pagerank = G.pagerank(weights='weight')

    # 近似 betweenness
    print(Fore.BLUE + f"近似计算 Betweenness，采样 k={approx_k} ...")
    if approx_k >= G.vcount():
        sampled_nodes = range(G.vcount())
    else:
        sampled_nodes = random.sample(range(G.vcount()), approx_k)
    betweenness = G.betweenness(vertices=sampled_nodes, directed=False, weights=None, cutoff=None)
    # 其他节点置0
    bet_dict = {G.vs[i]['name']: (betweenness[j] if i in sampled_nodes else 0)
                for j, i in enumerate(sampled_nodes)}
    # 所有节点保证有值
    metrics = {
        'degree': {G.vs[i]['name']: degree[i] for i in range(G.vcount())},
        'closeness': {G.vs[i]['name']: closeness[i] for i in range(G.vcount())},
        'pagerank': {G.vs[i]['name']: pagerank[i] for i in range(G.vcount())},
        'betweenness': bet_dict
    }
    return metrics

def save_results(G, metrics):
    # 保存 gexf
    gexf_path = os.path.join(RESULTS_DIR, "contributors_graph.gexf")
    G.write_gexf(gexf_path)
    print(Fore.GREEN + f"GEXF 保存完成: {gexf_path}")

    # 保存边列表
    edges_path = os.path.join(RESULTS_DIR, "contributors_edgelist.csv")
    with open(edges_path, 'w', encoding='utf-8') as fh:
        fh.write("source,target,weight\n")
        for e in G.es:
            fh.write(f"{G.vs[e.source]['name']},{G.vs[e.target]['name']},{e['weight']}\n")
    print(Fore.GREEN + f"Edgelist 保存完成: {edges_path}")

    # 保存节点指标
    metrics_path = os.path.join(RESULTS_DIR, "contributors_metrics.parquet")
    df_metrics = pd.DataFrame([
        {"Node": n,
         "Degree": metrics['degree'].get(n,0),
         "Betweenness": metrics['betweenness'].get(n,0),
         "Closeness": metrics['closeness'].get(n,0),
         "PageRank": metrics['pagerank'].get(n,0)}
        for n in G.vs['name']
    ])
    df_metrics.to_parquet(metrics_path, index=False)
    print(Fore.GREEN + f"节点指标保存完成: {metrics_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approx-betweenness", type=int, default=DEFAULT_APPROX_K,
                        help="采样节点数用于近似 betweenness，默认1000")
    args = parser.parse_args()

    df = read_data()
    edge_weights, node_set = build_contrib_edges(df, MIN_CONTRIBS_PER_REPO)
    G = build_igraph_graph(edge_weights, node_set)
    metrics = compute_metrics(G, approx_k=args.approx_betweenness)
    save_results(G, metrics)
    print(Fore.GREEN + "全部完成 ✅")

if __name__ == "__main__":
    main()
