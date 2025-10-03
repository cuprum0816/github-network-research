#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建贡献者-贡献者投影网络 (带权重的 Edge List + Node List)
输入：repo_to_contribs_top1000.parquet  (包含 Repo_ID, Contributors)
输出：
  - results/contributors_edges.parquet  (source, target, weight)
  - results/contributors_nodes.parquet  (node, degree, repos_count)
"""

import pandas as pd
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict, Counter
import os

INPUT_FILE = "repo_to_contribs_top1000.parquet"
EDGES_FILE = "results/contributors_edges.parquet"
NODES_FILE = "results/contributors_nodes.parquet"


def main():
    print("读取数据 ...")
    df = pd.read_parquet(INPUT_FILE)

    # 使用字典存储边权重，避免重复边
    edge_weights = defaultdict(int)
    # 记录每个节点参与的仓库数
    node_repos = defaultdict(set)

    print("构建贡献者-贡献者投影网络...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理仓库"):
        contribs = row["Contributors"]
        repo_id = row.get("Repo_ID", idx)  # 使用 Repo_ID 或索引

        # 去重并转换为字符串
        users = sorted(set(str(u) for u in contribs))

        # 记录每个用户参与的仓库
        for user in users:
            node_repos[user].add(repo_id)

        # 两两组合构建边，累加权重
        for u, v in combinations(users, 2):
            # 确保边的方向一致 (字典序小的在前)
            edge = (u, v) if u < v else (v, u)
            edge_weights[edge] += 1

    print(f"共生成 {len(edge_weights)} 条唯一边, {len(node_repos)} 个节点")

    # 构建边表（带权重）
    print("构建边表...")
    edges_data = [
        {"source": u, "target": v, "weight": w}
        for (u, v), w in edge_weights.items()
    ]
    edges_df = pd.DataFrame(edges_data)

    # 保存边表
    os.makedirs("results", exist_ok=True)
    edges_df.to_parquet(EDGES_FILE, index=False)
    print(f"边表已保存: {EDGES_FILE}")
    print(f"  - 平均权重: {edges_df['weight'].mean():.2f}")
    print(f"  - 最大权重: {edges_df['weight'].max()}")

    # 构建节点表（包含度和仓库数统计）
    print("构建节点表...")
    # 计算每个节点的度（边数）
    node_degrees = Counter()
    for u, v in edge_weights.keys():
        node_degrees[u] += 1
        node_degrees[v] += 1

    nodes_data = [
        {
            "node": node,
            "degree": node_degrees[node],
            "repos_count": len(repos)
        }
        for node, repos in node_repos.items()
    ]
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df = nodes_df.sort_values("degree", ascending=False)

    # 保存节点表
    nodes_df.to_parquet(NODES_FILE, index=False)
    print(f"节点表已保存: {NODES_FILE}")
    print(f"  - Top 5 节点 (按度排序):")
    print(nodes_df.head().to_string(index=False))

    # 输出统计信息
    print("\n网络统计:")
    print(f"  - 节点数: {len(nodes_df)}")
    print(f"  - 边数: {len(edges_df)}")
    print(f"  - 平均度: {nodes_df['degree'].mean():.2f}")
    print(f"  - 密度: {2 * len(edges_df) / (len(nodes_df) * (len(nodes_df) - 1)):.6f}")


if __name__ == "__main__":
    main()