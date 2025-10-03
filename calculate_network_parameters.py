#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的贡献者网络构建与分析流程
步骤1: 构建网络（边表+节点表）
步骤2: 计算网络指标（度中心性、介数中心性、接近中心性、PageRank等）
"""

import pandas as pd
import igraph as ig
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import os
import random

# ============ 配置参数 ============
INPUT_FILE = "repo_to_contribs_top1000.parquet"
EDGES_FILE = "results/contributors_edges.parquet"
NODES_FILE = "results/contributors_nodes.parquet"
METRICS_FILE = "results/contributors_metrics.csv"

# 指标计算采样参数（用于大规模网络）
BETWEENNESS_SAMPLE = 1000  # 介数中心性采样节点数，0表示全部计算
CLOSENESS_SAMPLE = 1000  # 接近中心性采样节点数，0表示全部计算
N_JOBS = cpu_count()  # 并行计算核心数


# ============ 步骤1: 构建网络 ============

def build_network():
    """构建贡献者协作网络"""
    print("\n" + "=" * 60)
    print("步骤1: 构建贡献者网络")
    print("=" * 60)

    print("读取数据 ...")
    df = pd.read_parquet(INPUT_FILE)

    # 使用字典存储边权重
    edge_weights = defaultdict(int)
    # 记录每个节点参与的仓库
    node_repos = defaultdict(set)

    print("构建贡献者-贡献者投影网络...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理仓库"):
        contribs = row["Contributors"]
        repo_id = row.get("Repo_ID", idx)

        # 去重并转换为字符串
        users = sorted(set(str(u) for u in contribs))

        # 记录每个用户参与的仓库
        for user in users:
            node_repos[user].add(repo_id)

        # 两两组合构建边，累加权重
        for u, v in combinations(users, 2):
            edge = (u, v) if u < v else (v, u)
            edge_weights[edge] += 1

    print(f"共生成 {len(edge_weights)} 条唯一边, {len(node_repos)} 个节点")

    # 构建边表
    print("保存边表...")
    edges_data = [
        {"source": u, "target": v, "weight": w}
        for (u, v), w in edge_weights.items()
    ]
    edges_df = pd.DataFrame(edges_data)

    os.makedirs("results", exist_ok=True)
    edges_df.to_parquet(EDGES_FILE, index=False)
    print(f"边表已保存: {EDGES_FILE}")
    print(f"  - 平均权重: {edges_df['weight'].mean():.2f}")
    print(f"  - 最大权重: {edges_df['weight'].max()}")

    # 构建节点表（初步统计）
    print("保存节点表...")
    node_degrees = Counter()
    for u, v in edge_weights.keys():
        node_degrees[u] += 1
        node_degrees[v] += 1

    nodes_data = [
        {
            "node": node,
            "initial_degree": node_degrees[node],
            "repos_count": len(repos)
        }
        for node, repos in node_repos.items()
    ]
    nodes_df = pd.DataFrame(nodes_data).sort_values("initial_degree", ascending=False)
    nodes_df.to_parquet(NODES_FILE, index=False)
    print(f"节点表已保存: {NODES_FILE}")

    # 输出基本统计
    print("\n网络基本统计:")
    print(f"  - 节点数: {len(nodes_df)}")
    print(f"  - 边数: {len(edges_df)}")
    print(f"  - 平均度: {nodes_df['initial_degree'].mean():.2f}")
    if len(nodes_df) > 1:
        density = 2 * len(edges_df) / (len(nodes_df) * (len(nodes_df) - 1))
        print(f"  - 网络密度: {density:.6f}")


# ============ 步骤2: 计算网络指标 ============

def read_graph():
    """读取网络数据并构建igraph图对象"""
    print("\n读取节点和边表...")
    nodes_df = pd.read_parquet(NODES_FILE)
    edges_df = pd.read_parquet(EDGES_FILE)

    print(f"  - 节点数: {len(nodes_df)}")
    print(f"  - 边数: {len(edges_df)}")

    nodes = nodes_df['node'].tolist()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # 构建边列表（整数索引）
    edges = []
    weights = []
    for _, row in edges_df.iterrows():
        src, tgt = row['source'], row['target']
        if src in node_to_idx and tgt in node_to_idx:
            edges.append((node_to_idx[src], node_to_idx[tgt]))
            weights.append(row.get('weight', 1))

    print(f"构建图对象...")
    g = ig.Graph(edges=edges, directed=False)
    g.vs["name"] = nodes
    g.es["weight"] = weights

    print(f"图构建完成: {len(g.vs)} 个节点, {len(g.es)} 条边")
    return g


def compute_basic_metrics(graph, closeness_sample):
    """计算基础网络指标"""
    print("\n计算基础指标...")

    # 度中心性
    print("  - 计算度中心性 (Degree)")
    degree = graph.degree()

    # 加权度中心性
    print("  - 计算加权度中心性 (Weighted Degree)")
    weighted_degree = graph.strength(weights="weight")

    # 接近中心性（可采样）
    n_vertices = len(graph.vs)
    if closeness_sample and closeness_sample < n_vertices:
        print(f"  - 计算接近中心性 (Closeness, 采样 {closeness_sample} 个节点)")
        sampled_indices = random.sample(range(n_vertices), closeness_sample)
        closeness = [0.0] * n_vertices
        sampled_closeness = graph.closeness(vertices=sampled_indices)
        for idx, value in zip(sampled_indices, sampled_closeness):
            closeness[idx] = value
    else:
        print("  - 计算接近中心性 (Closeness, 全部节点)")
        closeness = graph.closeness()

    # PageRank
    print("  - 计算 PageRank")
    pagerank = graph.pagerank(weights="weight")

    # 聚类系数
    print("  - 计算聚类系数 (Clustering Coefficient)")
    clustering = graph.transitivity_local_undirected()

    metrics_df = pd.DataFrame({
        "node": graph.vs["name"],
        "degree": degree,
        "weighted_degree": weighted_degree,
        "closeness": closeness,
        "pagerank": pagerank,
        "clustering_coefficient": clustering
    })

    return metrics_df


def betweenness_worker(args):
    """并行计算介数中心性的工作函数"""
    edges, weights, num_vertices, node_names, chunk_indices = args
    # 在每个进程中重建图
    g = ig.Graph(edges=edges, directed=False, n=num_vertices)
    g.vs["name"] = node_names
    g.es["weight"] = weights
    # 计算指定节点的介数中心性
    return g.betweenness(vertices=chunk_indices, weights="weight")


def compute_betweenness_parallel(graph, sample_size, n_jobs):
    """并行计算介数中心性"""
    print(f"\n计算介数中心性 (Betweenness)...")

    edges = [(e.source, e.target) for e in graph.es]
    weights = graph.es["weight"]
    num_vertices = len(graph.vs)
    node_names = graph.vs["name"]

    # 确定要计算的节点
    if sample_size and sample_size < num_vertices:
        print(f"  - 采样 {sample_size} 个节点")
        sampled_indices = random.sample(range(num_vertices), sample_size)
    else:
        print(f"  - 计算全部 {num_vertices} 个节点")
        sampled_indices = list(range(num_vertices))

    # 分块并行计算
    num_chunks = min(n_jobs * 4, len(sampled_indices))
    chunk_size = len(sampled_indices) // num_chunks + 1
    index_chunks = [sampled_indices[i:i + chunk_size]
                    for i in range(0, len(sampled_indices), chunk_size)]

    print(f"  - 分成 {len(index_chunks)} 个块，使用 {n_jobs} 个核心并行计算")

    args_list = [(edges, weights, num_vertices, node_names, chunk)
                 for chunk in index_chunks]

    betweenness_results = [0.0] * num_vertices

    with Pool(n_jobs) as pool:
        for chunk_idx, res in enumerate(tqdm(pool.imap(betweenness_worker, args_list),
                                             total=len(args_list),
                                             desc="  计算进度")):
            # 填充结果
            for local_idx, value in enumerate(res):
                global_idx = index_chunks[chunk_idx][local_idx]
                betweenness_results[global_idx] = value

    return betweenness_results


def analyze_network():
    """网络指标分析主流程"""
    print("\n" + "=" * 60)
    print("步骤2: 计算网络指标")
    print("=" * 60)

    # 读取图
    g = read_graph()

    # 计算基础指标
    metrics_df = compute_basic_metrics(g, CLOSENESS_SAMPLE)

    # 计算介数中心性（并行）
    betweenness = compute_betweenness_parallel(g, BETWEENNESS_SAMPLE, N_JOBS)
    metrics_df["betweenness"] = betweenness

    # 保存结果
    print(f"\n保存结果到 {METRICS_FILE} ...")
    metrics_df.to_csv(METRICS_FILE, index=False)
    print("✓ 保存完成")

    # 显示统计信息
    print("\n" + "=" * 60)
    print("网络指标统计摘要")
    print("=" * 60)
    print(metrics_df.describe())

    print("\nTop 10 节点 (按 PageRank 排序):")
    print(metrics_df.nlargest(10, 'pagerank')[['node', 'degree', 'pagerank', 'betweenness']])


# ============ 主函数 ============

def main():
    """主流程：构建网络 -> 计算指标"""
    print("\n" + "=" * 60)
    print("贡献者网络分析流程")
    print("=" * 60)

    # 检查是否需要重新构建网络
    if not os.path.exists(EDGES_FILE) or not os.path.exists(NODES_FILE):
        print("\n未找到网络文件，开始构建网络...")
        build_network()
    else:
        print("\n已找到网络文件，跳过构建步骤")
        print(f"  - {EDGES_FILE}")
        print(f"  - {NODES_FILE}")
        rebuild = input("是否重新构建网络? (y/N): ").strip().lower()
        if rebuild == 'y':
            build_network()

    # 计算网络指标
    analyze_network()

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - 边表: {EDGES_FILE}")
    print(f"  - 节点表: {NODES_FILE}")
    print(f"  - 指标表: {METRICS_FILE}")


if __name__ == "__main__":
    main()