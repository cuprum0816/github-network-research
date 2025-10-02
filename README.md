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

我们以相同的仓库为边，建立这些贡献者的社会网络。
~~~Python3

~~~

该程序以