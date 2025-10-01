import networkx as nx
import pandas as pd
from tqdm import tqdm  # 进度条
from colorama import init, Fore
import random
import scipy

init(autoreset=True)  # colorama 初始化

# 设置变量

data_path = "../data/repositories_withContributionCount.parquet"
SAMPLE_SIZE = 40000  # 采样率


# 构建集合函数

def build_repo_and_contributor_set(df):
    # 仓库 → 贡献者集合
    repo_to_contribs = {}
    print("正在预处理数据:")
    repo_groups = df.groupby("Repo_ID")["contributor_ID"]
    print("正在生成仓库 → 贡献者集合...")
    for repo, group in tqdm(repo_groups, total=repo_groups.ngroups):
        repo_to_contribs[repo] = set(group)
    print(f"完成，共 {len(repo_to_contribs)} 个仓库\n 正在保存:")
    pd.DataFrame([
        {"Repo_ID": repo, "Contributors": list(contribs)}
        for repo, contribs in repo_to_contribs.items()
    ]).to_parquet("repo_to_contribs.parquet", index=False)
    print(Fore.GREEN + "仓库集合保存完成")

    # 贡献者 → 仓库集合
    contrib_to_repos = {}
    print("正在生成贡献者 → 仓库集合...")
    contrib_groups = df.groupby("contributor_ID")["Repo_ID"]
    for contrib, group in tqdm(contrib_groups, total=contrib_groups.ngroups):
        contrib_to_repos[contrib] = set(group)
    print(f"完成，共 {len(contrib_to_repos)} 个贡献者\n正在保存:")
    pd.DataFrame([
        {"Contributor_ID": contrib, "Repos": list(repos)}
        for contrib, repos in contrib_to_repos.items()
    ]).to_parquet("contrib_to_repos.parquet", index=False)
    print(Fore.GREEN + "贡献者集合保存完成")



# 构建投影网络函数（采样）

def build_sampled_projection(node_dict, sample_size, desc="采样网络生成"):
    nodes = list(node_dict.keys())
    sampled_nodes = random.sample(nodes, min(sample_size, len(nodes)))
    G = nx.Graph()
    G.add_nodes_from(sampled_nodes)

    for i, n1 in enumerate(tqdm(sampled_nodes, desc=desc)):
        set1 = node_dict[n1]
        for n2 in sampled_nodes[i + 1:]:
            set2 = node_dict[n2]
            w = len(set1 & set2)
            if w > 0:
                G.add_edge(n1, n2, weight=w)
    return G


# 计算指标

def compute_network_metrics(G, desc="指标计算"):
    metrics = {}
    print(f"{desc} - degree...")
    metrics['degree'] = dict(G.degree())
    print(f"{desc} - betweenness...")
    metrics['betweenness'] = nx.betweenness_centrality(G)
    print(f"{desc} - closeness...")
    metrics['closeness'] = nx.closeness_centrality(G)
    print(f"{desc} - PageRank...")
    metrics['pagerank'] = nx.pagerank(G)
    return metrics

# 保存网络和指标
def save_network_and_metrics(G, metrics, prefix):
    nx.write_gexf(G, f"{prefix}_graph_sample.gexf")
    nx.write_edgelist(G, f"{prefix}_edgelist_sample.csv", delimiter=",", data=["weight"])
    pd.DataFrame([
        {"Node": n,
         "Degree": metrics['degree'][n],
         "Betweenness": metrics['betweenness'][n],
         "Closeness": metrics['closeness'][n],
         "PageRank": metrics['pagerank'][n]}
        for n in G.nodes()
    ]).to_parquet(f"{prefix}_metrics_sample.parquet", index=False)
    print(Fore.GREEN + f"{prefix} 网络和指标已保存")

# 主程序
if __name__ == "__main__":
    print("\nGitHub Social Network Research")
    print("https://github.com/cuprum0816\n")
    print(Fore.BLUE + "正在读取数据...", Fore.RESET + f"数据路径 {data_path}")
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(Fore.RED + "读取数据错误: FileNotFoundError, 程序即将退出。请更改data_path。")
        exit()
    print(Fore.GREEN + f"数据读取完成，共 {len(df)} 行。")

    print("请选择功能:")
    print("1. 构建贡献者集合与仓库集合")
    print("2. 开始构建网络（请先执行第一步）")
    case_select = input("请输入选项: ")

    if case_select == "1":
        build_repo_and_contributor_set(df)

    elif case_select == "2":
        # 加载之前保存的集合
        try:
            repo_df = pd.read_parquet("repo_to_contribs.parquet")
            contrib_df = pd.read_parquet("contrib_to_repos.parquet")
        except FileNotFoundError:
            print(Fore.RED + "请先运行选项1生成集合文件！")
            exit()

        # 转换成字典
        repo_to_contribs = {row['Repo_ID']: set(row['Contributors']) for _, row in
                            tqdm(repo_df.iterrows(), total=len(repo_df), desc="加载仓库集合")}
        contrib_to_repos = {row['Contributor_ID']: set(row['Repos']) for _, row in
                            tqdm(contrib_df.iterrows(), total=len(contrib_df), desc="加载贡献者集合")}

        # 构建采样网络
        G_repos_sample = build_sampled_projection(repo_to_contribs, SAMPLE_SIZE, desc="生成仓库-仓库采样网络")
        G_contrib_sample = build_sampled_projection(contrib_to_repos, SAMPLE_SIZE, desc="生成贡献者-贡献者采样网络")

        # 计算指标
        metrics_repos_sample = compute_network_metrics(G_repos_sample, desc="仓库采样网络")
        metrics_contrib_sample = compute_network_metrics(G_contrib_sample, desc="贡献者采样网络")

        # 保存
        save_network_and_metrics(G_repos_sample, metrics_repos_sample, "repos")
        save_network_and_metrics(G_contrib_sample, metrics_contrib_sample, "contributors")

    else:
        print(Fore.RED + "输入错误。")

