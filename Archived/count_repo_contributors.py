

import pandas as pd

file_path = "../data/contributors_withContributionCount.csv"  # 把这里换成你的文件名
df = pd.read_csv(file_path)
repo_counts = df['Repo_ID'].value_counts()
print("每个 Repo_ID 出现的次数：")
print(repo_counts)

# 可选：保存统计结果到新的 CSV 文件
repo_counts.to_csv("repo_counts.csv", header=['Count'])