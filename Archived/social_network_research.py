# social network research
# author: cuprum
import pandas as pd
# Load Data

repo_network_data = pd.read_parquet("repos_metrics_sample.parquet")
'''
该数据集是每个repo的各个指标。下面是一个实例：
        Node  Degree   Betweenness  Closeness  PageRank
0  424507010       1      0.000000   0.276858  0.000019
1  594155488     236  46699.466380   0.430289  0.000333
'''
repo_network_data.head()
print(repo_network_data.head())

# 我们需要搞清楚如何构建仓库好坏的指标

