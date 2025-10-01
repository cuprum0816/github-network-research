import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("../data/cleaned_file.csv")

# 保存为 Parquet 格式
df.to_parquet("repositories_withContributionCount.parquet", engine="pyarrow", index=False)
