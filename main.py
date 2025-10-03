from tqdm import tqdm
import pandas as pd
from colorama import init, Fore
df = pd.read_parquet("./data/repositories_withContributionCount_more_than_1.parquet")
CONTRIBUTORS = 250
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

df_stars = pd.read_csv("./data/repositories_star_count.csv")
df_contribs = pd.read_parquet("repo_to_contribs.parquet")
# 取 star 数前1000的 Repo
top1000 = df_stars.sort_values("star_count", ascending=False).head(CONTRIBUTORS)
df_filtered = df_contribs[df_contribs["Repo_ID"].isin(top1000["repo_id"])]
# 保存结果
df_filtered.to_parquet("repo_to_contribs_top1000.parquet", index=False)
print(f"筛选完成，共保留 {df_filtered['Repo_ID'].nunique()} 个仓库")
repos_top1000 = pd.read_parquet("repo_to_contribs_top1000.parquet")
contribs = pd.read_parquet("./data/repositories_withContributionCount_more_than_1.parquet")
filtered = contribs[contribs['Repo_ID'].isin(repos_top1000['Repo_ID'])]
filtered.to_parquet("contributors_in_top1000_repos.parquet", index=False)