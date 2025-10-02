# github-network-research

本项目旨在抽取github 2024年Star数量前1000的仓库里面的所有贡献者构建社会网络。

## 1. 数据预处理

### 数据获取
我们使用 BigQuery 中的 github-archive数据库，利用其2024年的数据，查询全年star数量前10000的仓库。

~~~SQL
SELECT
  repo.id AS repo_id,
  repo.name AS repo_name,
  COUNT(1) AS star_count
FROM
  `githubarchive.year.2024`
WHERE
  type = 'WatchEvent'
GROUP BY
  repo.id, repo.name
ORDER BY
  star_count DESC
LIMIT 10000;
~~~

我们得到了一张表，该表被保存为:

github-network-465507.github.repos_top_10000_in_2024

该表的格式为:

| repo_id | repo_name | star_count |
|---------|-----------|------------|
| 仓库唯一代码  | 仓库名称      | 仓库Star数量   |

我们接着获取这些仓库的活动，同样使用github-archive数据库。

~~~SQL
SELECT
  original.type AS Type,
  repos.repo_id AS Repo_ID,
  repos.repo_name AS Repo_Name,
  original.actor.id AS contributor_ID,
  original.actor.login AS contributor,
FROM
  `github-network-465507.github.repos_top_10000_in_2024` AS repos
JOIN
  `githubarchive.year.2024` AS original
ON
  repos.repo_id = original.repo.id
WHERE
  original.type IN (
    'PushEvent',
    'PullRequestEvent',
  );
~~~
我们只保留了改动代码的贡献（PushEvent和PullRequestEvent）。

为了极大程度的简化操作，我们将所有贡献计数并保存。
~~~SQL
SELECT
  Repo_ID,
  contributor_ID,
  contributor,
  COUNT(*) AS contribution_count
FROM `github-network-465507.github.EventList_withoutBots_Final`
GROUP BY Repo_ID, contributor_ID, contributor
ORDER BY contribution_count DESC;
~~~

该表的格式为：

| Repo_ID |contributor_ID| contributor | contribution_count |
|-------|--------|------|------------|
|仓库的唯一代码|贡献者的唯一代码| 贡献者昵称| 贡献者对仓库贡献的次数|

至此，我们已经获得了需要的数据。该数据被保存为**contributors_withContributionCount.csv**

### 数据清洗
在这一环节，我们旨在清除数据中存在的机器人。这些机器人的提交权重极高，会极大程度污染数据源。

~~~Python3
import pandas as pd
import re
# 读入 CSV
df = pd.read_csv("Archived/Contributor_withContributonCount.csv")

patterns = [
    r"bot",
    r"machine",
    r"automation",
    r"queue",
    r"testing",
    r"promoter",
    r"-ci$",
    r"\bci\b"
]

def is_bot(name):
    name_lower = str(name).lower()
    return any(re.search(p, name_lower) for p in patterns)

# 新增一列标记是否为机器人
df["is_bot"] = df["contributor"].apply(is_bot)

# 提取机器人名单
bots = df[df["is_bot"]]

# 按 contributor 分组，累加 contribution_count
bots_grouped = bots.groupby("contributor")["contribution_count"].sum().reset_index()

# 按贡献次数从高到低排序
bots_grouped_sorted = bots_grouped.sort_values(by="contribution_count", ascending=False)

# 打印结果
print("机器人名单及贡献次数（按贡献从高到低）：")
for _, row in bots_grouped_sorted.iterrows():
    print(f"{row['contributor']}")

~~~
我们因此获得了机器人的名单(bots.txt)利用 bots.txt，我们进行二次筛选，并删除所有机器人。
~~~Python3
import pandas as pd

df = pd.read_csv("contributors_withContributionCount.csv)

# bots.txt，每行一个名字
with open("../data/bots.txt", "r", encoding="utf-8") as f:
    bot_list = [line.strip() for line in f if line.strip()]
df_cleaned = df[~df["contributor"].isin(bot_list)]

df_cleaned.to_csv("cleaned_contributors.csv", index=False)

print(f"清洗完成！原始数据 {len(df)} 行，删除机器人 {len(df) - len(df_cleaned)} 行，剩下 {len(df_cleaned)} 行。")
~~~

为了更好处理数据，我们将原始文件保存为repositories_withContributionCount.parquet
~~~Python3
import pandas as pd
df = pd.read_csv("cleaned_contributors.csv")
df.to_parquet("repositories_withContributionCount.parquet", engine="pyarrow", index=False)
~~~

## 2. 社会网络绘制

### 贡献者筛选
为了更好的处理关系，我们选取前1000的贡献者目录。

~~~Python3
import tqdm
import pandas as pd

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
    
df_stars = pd.read_csv("./data/repositories_star_count.csv") # 假设这张表有 Repo_ID, Stars
df_contribs = pd.read_parquet("./data/repo_to_contribs.parquet")
# 取 star 数前1000的 Repo
top1000 = df_stars.sort_values("star_count", ascending=False).head(1000)
# 只保留这1000个 repo 的贡献者信息
df_filtered = df_contribs[df_contribs["Repo_ID"].isin(top1000["repo_id"])]
# 保存结果
df_filtered.to_parquet("repo_to_contribs_top1000.parquet", index=False)
print(f"筛选完成，共保留 {df_filtered['Repo_ID'].nunique()} 个仓库")
~~~
保存完毕的是repo_to_contribs_top1000.parquet。该文件保存了star数量前1000的仓库的贡献者集合。

接下来，我们获取这些贡献者的贡献：

~~~Python3
import pandas as pd
repos_top1000 = pd.read_parquet("repo_to_contribs_top1000.parquet")
contribs = pd.read_parquet("repositories_withContributionCount.parquet")  
filtered = contribs[contribs['Repo_ID'].isin(repos_top1000['Repo_ID'])]
filtered.to_parquet("contributors_in_top1000_repos.parquet", index=False)
~~~
这段代码获取了前1000仓库中所有贡献者的贡献情况。我们保留该文件作为备用。

| Repo_ID |contributor_ID| contributor | contribution_count |
|-------|--------|------|------------|
|仓库的唯一代码|贡献者的唯一代码| 贡献者昵称| 贡献者对仓库贡献的次数|

我们使用数据集repo_to_contribs_top1000.parquet，构建最终的社会网络。

### 贡献者-贡献者社会网络构建

我们以相同的仓库为边，建立这些贡献者的社会网络。首先，我们画图：
~~~Python3

import os
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import igraph as ig
from colorama import init, Fore
from collections import defaultdict

init(autoreset=True)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE = "./repo_to_contribs_top1000.parquet"
MIN_CONTRIBS_PER_REPO = 2
EDGE_CSV = os.path.join(RESULTS_DIR, "contributors_edgelist.csv")
CHUNK_SIZE = 1_000_000  # 每100万条边写一次

# ------------------ 读取数据 ------------------
def read_data():
    print(Fore.BLUE + f"读取数据: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    if 'Repo_ID' not in df.columns or 'Contributors' not in df.columns:
        raise ValueError("文件必须包含 'Repo_ID' 和 'Contributors' 列")
    print(Fore.GREEN + f"数据读取完成，共 {len(df)} 行")
    return df

# ------------------ 构建边 CSV（分块） ------------------
def build_edge_csv(df):
    edge_list = []
    print(Fore.BLUE + "构建边列表（分块写入 CSV）...")
    if os.path.exists(EDGE_CSV):
        os.remove(EDGE_CSV)

    node_set = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="仓库处理"):
        contribs = row['Contributors']
        if len(contribs) < MIN_CONTRIBS_PER_REPO:
            continue
        node_set.update(contribs)
        for a, b in combinations(contribs, 2):
            if a < b:
                edge_list.append((a, b, 1))
            else:
                edge_list.append((b, a, 1))
        if len(edge_list) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(edge_list, columns=['source','target','weight'])
            df_chunk.to_csv(EDGE_CSV, mode='a', index=False, header=not os.path.exists(EDGE_CSV))
            edge_list = []

    # 最后剩余的写入
    if edge_list:
        df_chunk = pd.DataFrame(edge_list, columns=['source','target','weight'])
        df_chunk.to_csv(EDGE_CSV, mode='a', index=False, header=not os.path.exists(EDGE_CSV))

    print(Fore.GREEN + f"边列表 CSV 保存完成: {EDGE_CSV}")
    return node_set

# ------------------ 构建 igraph 图 ------------------
def build_graph_from_csv():
    print(Fore.BLUE + "从 CSV 构建 igraph 图...")
    df_edges = pd.read_csv(EDGE_CSV)
    # 聚合边权
    df_agg = df_edges.groupby(['source','target'], as_index=False)['weight'].sum()
    G = ig.Graph.TupleList(df_agg.itertuples(index=False), weights=True, directed=False)
    print(Fore.GREEN + f"igraph 图构建完成: 节点数 {G.vcount()}, 边数 {G.ecount()}")
    return G

# ------------------ 保存图 ------------------
def save_graph(G):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    graphml_path = os.path.join(RESULTS_DIR, "contributors_graph.graphml")
    G.write_graphml(graphml_path)
    print(Fore.GREEN + f"GraphML 保存完成: {graphml_path}")
    pickle_path = os.path.join(RESULTS_DIR, "contributors_graph.pickle")
    G.write_pickle(pickle_path)
    print(Fore.GREEN + f"Pickle 保存完成: {pickle_path}")
    try:
        import networkx as nx
        from igraph import Graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(G.vs['name'])
        nx_graph.add_weighted_edges_from([
            (G.vs[e.source]['name'], G.vs[e.target]['name'], e['weight'])
            for e in G.es
        ])
        gexf_path = os.path.join(RESULTS_DIR, "contributors_graph.gexf")
        nx.write_gexf(nx_graph, gexf_path)
        print(Fore.GREEN + f"GEXF 保存完成: {gexf_path}")
    except ImportError:
        print(Fore.YELLOW + "未安装 networkx，跳过 GEXF 导出。")

# ------------------ 主函数 ------------------
def main():
    df = read_data()
    node_set = build_edge_csv(df)
    G = build_graph_from_csv()
    save_graph(G)
    print(Fore.GREEN + "图构建完成 ✅")

if __name__ == "__main__":
    main()

~~~
contributors_edgelist是边列表。

基于绘制完毕的igraph图（contributors_graph.graphml）,我们计算贡献者的各项指标。
