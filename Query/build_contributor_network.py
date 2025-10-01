# 该程序旨在构建每个仓库下的贡献者网络指标。
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 读取数据
df = pd.read_parquet("../data/repo_to_contribs.parquet")


def compute_repo_metrics(row):
    repo_id = row['Repo_ID']
    contribs = row['Contributors']

    if len(contribs) < 2:
        # 单人仓库直接返回0
        return [{"Repo_ID": repo_id, "Contributor_ID": contribs[0],
                 "Degree": 0, "Betweenness": 0, "Closeness": 0, "PageRank": 0}] if contribs else []

    # 构建全连接网络
    G = nx.Graph()
    G.add_nodes_from(contribs)
    for i in range(len(contribs)):
        for j in range(i + 1, len(contribs)):
            G.add_edge(contribs[i], contribs[j], weight=1)

    # 计算指标（大仓库可近似计算betweenness）
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, k=min(100, len(contribs)))  # 近似
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)

    results = []
    for c in contribs:
        results.append({
            "Repo_ID": repo_id,
            "Contributor_ID": c,
            "Degree": degree.get(c, 0),
            "Betweenness": betweenness.get(c, 0),
            "Closeness": closeness.get(c, 0),
            "PageRank": pagerank.get(c, 0)
        })
    return results


if __name__ == "__main__":
    n_cores = max(1, cpu_count() - 3)  # 保留一个核心给系统
    print(f"使用 {n_cores} 个核心计算...")

    rows = [row for _, row in df.iterrows()]
    results = []

    with Pool(n_cores) as pool:
        for repo_metrics in tqdm(pool.imap(compute_repo_metrics, rows), total=len(rows)):
            results.extend(repo_metrics)

    # 保存结果
    df_metrics = pd.DataFrame(results)
    df_metrics.to_parquet("repo_contributor_metrics_parallel.parquet", index=False)
    print("并行计算完成！")
