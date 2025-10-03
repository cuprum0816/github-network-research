#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建贡献者-贡献者投影网络 (Edge List + Node List)
输入：repo_to_contribs_top1000.parquet  (包含 Repo_ID, Contributors)
输出：
  - results/contributors_edges.parquet
  - results/contributors_nodes.parquet
"""

import pandas as pd
from tqdm import tqdm
from itertools import combinations
import os

INPUT_FILE = "repo_to_contribs_top1000.parquet"
EDGES_FILE = "results/contributors_edges.parquet"
NODES_FILE = "results/contributors_nodes.parquet"

def main():
    print("读取数据 ...")
    df = pd.read_parquet(INPUT_FILE)

    edges = []
    node_set = set()

    print("构建贡献者-贡献者投影网络权重...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        contribs = row["Contributors"]
        users = [str(u) for u in contribs]

        node_set.update(users)

        # 两两组合构建边
        for u, v in combinations(sorted(set(users)), 2):
            edges.append((u, v))

    print(f"共生成 {len(edges)} 条边, {len(node_set)} 个节点")

    # 保存边表
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    os.makedirs("results", exist_ok=True)
    edges_df.to_parquet(EDGES_FILE, index=False)
    print(f"边表已保存: {EDGES_FILE}")

    # 保存节点表
    nodes_df = pd.DataFrame({"node": list(node_set)})
    nodes_df.to_parquet(NODES_FILE, index=False)
    print(f"节点表已保存: {NODES_FILE}")

if __name__ == "__main__":
    main()
