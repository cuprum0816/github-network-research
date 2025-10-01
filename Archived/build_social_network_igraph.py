import pandas as pd
import igraph as ig
from tqdm import tqdm
from colorama import init, Fore
import random
import pickle

init(autoreset=True)

data_path = "../data/repositories_withContributionCount.parquet"
SAMPLE_SIZE = 20000  # 可以改成全量或更大

# 构建集合
def build_repo_and_contributor_set(df):
    print("正在生成仓库 → 贡献者集合...")
    repo_to_contribs = {}
    for repo, group in tqdm(df.groupby("Repo_ID")["contributor_ID"], total=df.groupby("Repo_ID")["contributor_ID"].ngroups):
        repo_to_contribs[repo] = set(group)
    pd.to_pickle(repo_to_contribs, "../repo_to_contribs.pkl")
    print(Fore.GREEN + "仓库集合保存完成")

    print("正在生成贡献者 → 仓库集合...")
    contrib_to_repos = {}
    for contrib, group in tqdm(df.groupby("contributor_ID")["Repo_ID"], total=df.groupby("contributor_ID")["Repo_ID"].ngroups):
        contrib_to_repos[contrib] = set(group)
    pd.to_pickle(contrib_to_repos, "../contrib_to_repos.pkl")
    print(Fore.GREEN + "贡献者集合保存完成")


# 构建投影网络
def build_sampled_projection_igraph(node_dict, sample_size, desc="采样网络生成"):
    nodes = list(node_dict.keys())
    sampled_nodes = random.sample(nodes, min(sample_size, len(nodes)))
    g = ig.Graph()
    g.add_vertices(len(sampled_nodes))
    node_index = {n: i for i, n in enumerate(sampled_nodes)}

    edges = []
    weights = []

    for i, n1 in enumerate(tqdm(sampled_nodes, desc=desc)):
        set1 = node_dict[n1]
        for n2 in sampled_nodes[i+1:]:
            set2 = node_dict[n2]
            w = len(set1 & set2)
            if w > 0:
                edges.append((node_index[n1], node_index[n2]))
                weights.append(w)

    g.add_edges(edges)
    g.es['weight'] = weights
    g.vs['name'] = sampled_nodes
    return g


# 计算指标
def compute_network_metrics_igraph(g, desc="指标计算"):
    metrics = {}
    print(Fore.BLUE + f"{desc} - 计算度...")
    metrics['degree'] = g.degree()
    print(Fore.BLUE + f"{desc} - 计算介数中心性...")
    metrics['betweenness'] = g.betweenness(directed=False, weights=None)
    print(Fore.BLUE + f"{desc} - 计算接近中心性...")
    metrics['closeness'] = g.closeness()
    print(Fore.BLUE + f"{desc} - 计算PageRank...")
    metrics['pagerank'] = g.pagerank()
    return metrics


# 保存网络和指标
def save_network_and_metrics_igraph(g, metrics, prefix):
    g.write_gml(f"{prefix}_graph_sample.gml")  # igraph推荐gml或graphml
    pd.DataFrame({
        "Node": g.vs['name'],
        "Degree": metrics['degree'],
        "Betweenness": metrics['betweenness'],
        "Closeness": metrics['closeness'],
        "PageRank": metrics['pagerank']
    }).to_parquet(f"{prefix}_metrics_sample.parquet", index=False)
    print(Fore.GREEN + f"{prefix} 网络和指标已保存")


if __name__ == "__main__":
    print("\nGitHub Social Network Research")
    print("https://github.com/cuprum0816\n")
    print(Fore.BLUE + "正在读取数据...", Fore.RESET + f"数据路径 {data_path}")
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(Fore.RED + "读取数据错误: FileNotFoundError, 程序即将退出。")
        exit()
    print(Fore.GREEN + f"数据读取完成，共 {len(df)} 行。")

    print("请选择功能:")
    print("1. 构建贡献者集合与仓库集合")
    print("2. 构建采样网络并计算指标")
    case_select = input("请输入选项: ")

    if case_select == "1":
        build_repo_and_contributor_set(df)

    elif case_select == "2":
        # 加载之前保存的集合
        try:
            repo_to_contribs = pd.read_pickle("../repo_to_contribs.pkl")
            contrib_to_repos = pd.read_pickle("../contrib_to_repos.pkl")
        except FileNotFoundError:
            print(Fore.RED + "请先运行选项1生成集合文件！")
            exit()

        # 构建采样网络
        G_repos_sample = build_sampled_projection_igraph(repo_to_contribs, SAMPLE_SIZE, desc="生成仓库-仓库采样网络")
        G_contrib_sample = build_sampled_projection_igraph(contrib_to_repos, SAMPLE_SIZE, desc="生成贡献者-贡献者采样网络")

        # 计算指标
        metrics_repos_sample = compute_network_metrics_igraph(G_repos_sample, desc="仓库采样网络")
        metrics_contrib_sample = compute_network_metrics_igraph(G_contrib_sample, desc="贡献者采样网络")

        # 保存
        save_network_and_metrics_igraph(G_repos_sample, metrics_repos_sample, "repos")
        save_network_and_metrics_igraph(G_contrib_sample, metrics_contrib_sample, "contributors")

    else:
        print(Fore.RED + "输入错误。")
