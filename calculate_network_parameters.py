#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import igraph as ig
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

EDGES_FILE = "results/contributors_edges.parquet"
NODES_FILE = "results/contributors_nodes.parquet"
N_JOBS = cpu_count()  # 使用所有可用 CPU 核心
BETWEENNESS_SAMPLE = 1000  # 可调，越大越准
CLOSENESS_SAMPLE = 1000  # Closeness 采样数量


def read_graph(edges_file, nodes_file):
    print("读取节点和边表 ...")
    nodes_df = pd.read_parquet(nodes_file)
    edges_df = pd.read_parquet(edges_file)

    # 所有节点列表
    nodes = nodes_df['node'].tolist()

    # 构建节点 ID -> 索引 映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # 用整数索引替换 edges
    edges = [(node_to_idx[src], node_to_idx[tgt])
             for src, tgt in zip(edges_df['source'], edges_df['target'])
             if src in node_to_idx and tgt in node_to_idx]

    print(f"构建 igraph 图: {len(nodes)} 个节点, {len(edges)} 条边 ...")
    g = ig.Graph(edges=edges, directed=False)
    g.vs["name"] = nodes
    print(f"图构建完成: 节点数 {len(g.vs)}, 边数 {len(g.es)}")
    return g, nodes


def compute_basic_metrics(graph, closeness_sample):
    print("计算 Degree ...")
    degree = graph.degree()

    print(f"计算 Closeness (采样 {closeness_sample}) ...")
    if closeness_sample < len(graph.vs):
        import random
        sampled_indices = random.sample(range(len(graph.vs)), closeness_sample)
        closeness = [0.0] * len(graph.vs)
        sampled_closeness = graph.closeness(vertices=sampled_indices)
        for idx, value in zip(sampled_indices, sampled_closeness):
            closeness[idx] = value
    else:
        closeness = graph.closeness()

    print("计算 PageRank ...")
    pagerank = graph.pagerank()

    metrics_df = pd.DataFrame({
        "node": graph.vs["name"],
        "degree": degree,
        "closeness": closeness,
        "pagerank": pagerank
    })
    return metrics_df


def betweenness_worker(args):
    """Worker function for parallel betweenness computation"""
    edges, num_vertices, node_names, chunk_indices = args
    # 在每个进程中重建图（避免序列化整个图对象）
    g = ig.Graph(edges=edges, directed=False, n=num_vertices)
    g.vs["name"] = node_names
    # 只计算这个chunk的节点
    return g.betweenness(vertices=chunk_indices)


if __name__ == "__main__":
    g, nodes = read_graph(EDGES_FILE, NODES_FILE)

    # 计算 Degree / Closeness / PageRank
    metrics_df = compute_basic_metrics(g, CLOSENESS_SAMPLE)

    # 分块计算 Betweenness
    print(f"分块多核计算 Betweenness (采样 {BETWEENNESS_SAMPLE}, 多核 {N_JOBS}) ...")

    # 准备图的基本数据用于多进程
    edges = [(e.source, e.target) for e in g.es]
    num_vertices = len(g.vs)
    node_names = g.vs["name"]

    # 根据采样数量选择节点
    if BETWEENNESS_SAMPLE < len(nodes):
        import random

        sampled_indices = random.sample(range(len(nodes)), BETWEENNESS_SAMPLE)
    else:
        sampled_indices = list(range(len(nodes)))

    # 分更多的块以提高并行效率和进度可见性
    num_chunks = min(N_JOBS * 4, len(sampled_indices))  # 分成 CPU核心数 x 4 个块
    chunk_size = len(sampled_indices) // num_chunks + 1
    index_chunks = [sampled_indices[i:i + chunk_size]
                    for i in range(0, len(sampled_indices), chunk_size)]

    print(f"将 {len(sampled_indices)} 个节点分成 {len(index_chunks)} 个块进行计算 ...")

    args_list = [(edges, num_vertices, node_names, chunk) for chunk in index_chunks]

    # 初始化 betweenness 为 0
    betweenness_results = [0.0] * len(nodes)

    with Pool(N_JOBS) as pool:
        for chunk_idx, res in enumerate(tqdm(pool.imap(betweenness_worker, args_list),
                                             total=len(args_list),
                                             desc="计算 Betweenness")):
            # 将结果填充到对应位置
            for local_idx, value in enumerate(res):
                global_idx = index_chunks[chunk_idx][local_idx]
                betweenness_results[global_idx] = value

    metrics_df["betweenness"] = betweenness_results

    # 保存结果
    print("保存结果 ...")
    metrics_df.to_csv("results/contributors_metrics.csv", index=False)
    print("计算完成，结果已保存为 results/contributors_metrics.csv")

    # 显示统计信息
    print("\n指标统计:")
    print(metrics_df.describe())