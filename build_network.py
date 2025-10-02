#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于保存图计算网络指标（大图安全版，带功能栏）
输入：results/contributors_graph.pickle
输出：results/contributors_metrics.parquet
"""

import os
import igraph as ig
import pandas as pd
from tqdm import tqdm
import random
from colorama import init, Fore
from concurrent.futures import ProcessPoolExecutor, as_completed

init(autoreset=True)

RESULTS_DIR = "./results"
DEFAULT_APPROX_K = 2000
DEFAULT_N_JOBS = 16


# ------------------ 单节点计算函数 ------------------
def compute_closeness_single(i, G):
    return G.vs[i]['name'], G.closeness(vertices=[i])[0]


def compute_betweenness_single(i, G):
    return G.vs[i]['name'], G.betweenness(vertices=[i], directed=False, weights=None, cutoff=None)[0]


# ------------------ 多核近似计算 ------------------
def compute_closeness_parallel(G, approx_k, n_jobs):
    if approx_k >= G.vcount():
        sampled_nodes = list(range(G.vcount()))
    else:
        sampled_nodes = random.sample(range(G.vcount()), approx_k)

    closeness_dict = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(compute_closeness_single, i, G): i for i in sampled_nodes}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Closeness"):
            node, value = f.result()
            closeness_dict[node] = value

    closeness = {G.vs[i]['name']: closeness_dict.get(G.vs[i]['name'], 0) for i in range(G.vcount())}
    return closeness


def compute_betweenness_parallel(G, approx_k, n_jobs):
    if approx_k >= G.vcount():
        sampled_nodes = list(range(G.vcount()))
    else:
        sampled_nodes = random.sample(range(G.vcount()), approx_k)

    betweenness_dict = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(compute_betweenness_single, i, G): i for i in sampled_nodes}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Betweenness"):
            node, value = f.result()
            betweenness_dict[node] = value

    for i in range(G.vcount()):
        if G.vs[i]['name'] not in betweenness_dict:
            betweenness_dict[G.vs[i]['name']] = 0

    return betweenness_dict


# ------------------ 保存指标 ------------------
def save_metrics(metrics, filename="contributors_metrics.parquet"):
    metrics_path = os.path.join(RESULTS_DIR, filename)
    df = pd.DataFrame(metrics)
    df.to_parquet(metrics_path, index=False)
    print(Fore.GREEN + f"✅ 指标保存完成: {metrics_path}")


# ------------------ 主函数 ------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-file", type=str,
                        default=os.path.join(RESULTS_DIR, "contributors_graph.pickle"),
                        help="保存的图文件 (.pickle 或 .graphml)")
    parser.add_argument("--approx-k", type=int, default=DEFAULT_APPROX_K,
                        help="近似采样节点数")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                        help="并行核数")
    args = parser.parse_args()

    # 加载图
    if args.graph_file.endswith(".pickle"):
        G = ig.Graph.Read_Pickle(args.graph_file)
    elif args.graph_file.endswith(".graphml"):
        G = ig.Graph.Read_GraphML(args.graph_file)
    else:
        raise ValueError("只支持 pickle 或 graphml 文件")
    print(Fore.GREEN + f"图加载完成: 节点数 {G.vcount()}, 边数 {G.ecount()}")


    metrics = {"Node": G.vs['name']}
    print(Fore.BLUE + "计算 Degree ...")
    metrics["Degree"] = G.degree()

    print(Fore.BLUE + f"并行近似计算 Closeness (cores={args.n_jobs}) ...")
    metrics["Closeness"] = list(compute_closeness_parallel(G, args.approx_k, args.n_jobs).values())

    print(Fore.BLUE + f"并行近似计算 Betweenness (cores={args.n_jobs}) ...")
    metrics["Betweenness"] = list(compute_betweenness_parallel(G, args.approx_k, args.n_jobs).values())

    print(Fore.BLUE + "计算 PageRank ...")
    metrics["PageRank"] = G.pagerank(weights='weight')

    save_metrics(metrics)


if __name__ == "__main__":
    main()
