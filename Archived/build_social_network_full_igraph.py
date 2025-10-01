import pandas as pd
import igraph as ig
import random
from tqdm import tqdm
from colorama import init, Fore
import multiprocessing as mp

init(autoreset=True)

# -----------------------------
# 配置
# -----------------------------
DATA_PATH = "../data/repositories_withContributionCount.parquet"
REPO_TO_CONTRIBS_FILE = "repo_to_contribs.parquet"
CONTRIB_TO_REPOS_FILE = "contrib_to_repos.parquet"
BETWEENNESS_CUTOFF = 3  # 近似介数中心性截断
N_PROCESS = 24  # CPU 核数

# -----------------------------
# 构建集合
# -----------------------------
def build_repo_and_contributor_set(df):
    # 仓库 → 贡献者集合
    print("生成仓库 → 贡献者集合...")
    repo_to_contribs = {}
    for repo, group in tqdm(df.groupby("Repo_ID")["contributor_ID"], total=df['Repo_ID'].nunique()):
        repo_to_contribs[repo] = set(group)
    pd.DataFrame([{"Repo_ID": k, "Contributors": list(v)} for k, v in repo_to_contribs.items()]).to_parquet(REPO_TO_CONTRIBS_FILE, index=False)
    print(Fore.GREEN + f"仓库集合保存完成: {REPO_TO_CONTRIBS_FILE}")

    # 贡献者 → 仓库集合
    print("生成贡献者 → 仓库集合...")
    contrib_to_repos = {}
    for contrib, group in tqdm(df.groupby("contributor_ID")["Repo_ID"], total=df['contributor_ID'].nunique()):
        contrib_to_repos[contrib] = set(group)
    pd.DataFrame([{"Contributor_ID": k, "Repos": list(v)} for k, v in contrib_to_repos.items()]).to_parquet(CONTRIB_TO_REPOS_FILE, index=False)
    print(Fore.GREEN + f"贡献者集合保存完成: {CONTRIB_TO_REPOS_FILE}")

    return repo_to_contribs, contrib_to_repos

# -----------------------------
# 构建投影网络
# -----------------------------
def build_full_projection(node_dict, desc="生成网络"):
    print(desc)
    g = ig.Graph()
    nodes = list(node_dict.keys())
    g.add_vertices(nodes)
    edges = []
    weights = []

    for i, n1 in enumerate(tqdm(nodes, desc=desc)):
        set1 = node_dict[n1]
        for n2 in nodes[i + 1:]:
            set2 = node_dict[n2]
            w = len(set1 & set2)
            if w > 0:
                edges.append((n1, n2))
                weights.append(w)

    g.add_edges(edges)
    g.es['weight'] = weights
    print(Fore.GREEN + f"网络生成完成: {g.vcount()} 节点, {g.ecount()} 边")
    return g

# -----------------------------
# 计算指标
# -----------------------------
def compute_metrics(g):
    print("计算度中心性...")
    degree = g.degree()
    print("计算近似介数中心性...")
    betweenness = g.betweenness(directed=False, cutoff=BETWEENNESS_CUTOFF, weights=None)
    print("计算接近中心性...")
    closeness = g.closeness()
    print("计算PageRank...")
    pagerank = g.pagerank()
    return {
        'degree': degree,
        'betweenness': betweenness,
        'closeness': closeness,
        'pagerank': pagerank
    }

# -----------------------------
# 保存网络和指标
# -----------------------------
def save_network_and_metrics(g, metrics, prefix):
    g.write_gexf(f"{prefix}_graph.gexf")
    g.write_edgelist(f"{prefix}_edgelist.csv", delimiter=",", weights=True)
    df = pd.DataFrame({
        "Node": g.vs['name'],
        "Degree": metrics['degree'],
        "Betweenness": metrics['betweenness'],
        "Closeness": metrics['closeness'],
        "PageRank": metrics['pagerank']
    })
    df.to_parquet(f"{prefix}_metrics.parquet", index=False)
    print(Fore.GREEN + f"{prefix} 网络和指标已保存")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    print("\nGitHub Social Network Research\n")
    print(Fore.BLUE + f"正在读取数据: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(Fore.GREEN + f"数据读取完成, {len(df)} 行")

    # 构建集合
    repo_to_contribs, contrib_to_repos = build_repo_and_contributor_set(df)

    # 构建全量网络
    G_repos = build_full_projection(repo_to_contribs, desc="生成仓库-仓库网络")
    G_contribs = build_full_projection(contrib_to_repos, desc="生成贡献者-贡献者网络")

    # 计算指标
    metrics_repos = compute_metrics(G_repos)
    metrics_contribs = compute_metrics(G_contribs)

    # 保存
    save_network_and_metrics(G_repos, metrics_repos, "repos")
    save_network_and_metrics(G_contribs, metrics_contribs, "contributors")
