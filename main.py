#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查贡献者网络数据的合理性
分析仓库规模分布和边生成情况
"""

import pandas as pd
import numpy as np
from collections import Counter

INPUT_FILE = "repo_to_contribs_top1000.parquet"
EDGES_FILE = "results/contributors_edges.parquet"


def analyze_repo_distribution():
    """分析仓库的贡献者分布"""
    print("=" * 60)
    print("仓库贡献者分布分析")
    print("=" * 60)

    df = pd.read_parquet(INPUT_FILE)

    # 统计每个仓库的贡献者数量
    repo_sizes = []
    for _, row in df.iterrows():
        contribs = row["Contributors"]
        unique_contribs = len(set(str(u) for u in contribs))
        repo_sizes.append(unique_contribs)

    repo_sizes = np.array(repo_sizes)

    print(f"\n仓库总数: {len(df)}")
    print(f"\n贡献者数量统计:")
    print(f"  平均值: {repo_sizes.mean():.1f}")
    print(f"  中位数: {np.median(repo_sizes):.1f}")
    print(f"  最小值: {repo_sizes.min()}")
    print(f"  最大值: {repo_sizes.max()}")
    print(f"  标准差: {repo_sizes.std():.1f}")

    # 分位数
    print(f"\n分位数分布:")
    for q in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(repo_sizes, q)
        print(f"  {q}%: {val:.0f}")

    # 理论边数计算
    print(f"\n理论最大边数计算:")
    theoretical_edges = sum([n * (n - 1) / 2 for n in repo_sizes])
    print(f"  如果每个仓库内所有人两两连边: {theoretical_edges:,.0f} 条")

    # 按规模分组
    print(f"\n仓库规模分组:")
    bins = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['1-10人', '11-50人', '51-100人', '101-200人',
              '201-500人', '501-1000人', '1000+人']

    repo_df = pd.DataFrame({'size': repo_sizes})
    repo_df['group'] = pd.cut(repo_df['size'], bins=bins, labels=labels)

    group_stats = repo_df.groupby('group', observed=True).agg({
        'size': ['count', 'mean', 'sum']
    }).round(1)

    print(group_stats)

    # 计算每组理论贡献的边数
    print(f"\n各规模组理论边数贡献:")
    for label in labels:
        group_repos = repo_df[repo_df['group'] == label]
        if len(group_repos) > 0:
            edges = sum([n * (n - 1) / 2 for n in group_repos['size']])
            print(f"  {label:15s}: {len(group_repos):4d} 个仓库 → {edges:12,.0f} 条边")

    return repo_sizes


def analyze_actual_network():
    """分析实际生成的网络"""
    print("\n" + "=" * 60)
    print("实际网络分析")
    print("=" * 60)

    edges_df = pd.read_parquet(EDGES_FILE)

    print(f"\n实际边数: {len(edges_df):,}")
    print(f"实际节点数: {len(set(edges_df['source']) | set(edges_df['target'])):,}")

    # 权重分布
    print(f"\n边权重分布:")
    print(f"  平均权重: {edges_df['weight'].mean():.2f}")
    print(f"  中位数权重: {edges_df['weight'].median():.0f}")
    print(f"  最大权重: {edges_df['weight'].max()}")

    # 权重分位数
    print(f"\n权重分位数:")
    for q in [50, 75, 90, 95, 99]:
        val = edges_df['weight'].quantile(q / 100)
        print(f"  {q}%: {val:.0f}")

    # 高权重边
    print(f"\n高权重边 (weight >= 10):")
    high_weight = edges_df[edges_df['weight'] >= 10]
    print(f"  数量: {len(high_weight):,} ({len(high_weight) / len(edges_df) * 100:.1f}%)")

    print(f"\nTop 10 高权重边:")
    print(high_weight.nlargest(10, 'weight'))


def check_duplicate_contributors():
    """检查是否有重复计算的贡献者"""
    print("\n" + "=" * 60)
    print("贡献者重复性检查")
    print("=" * 60)

    df = pd.read_parquet(INPUT_FILE)

    # 统计每个贡献者出现在多少个仓库
    contributor_repos = Counter()
    total_contributions = 0

    for _, row in df.iterrows():
        contribs = [str(u) for u in row["Contributors"]]
        total_contributions += len(contribs)
        for user in set(contribs):
            contributor_repos[user] += 1

    print(f"\n总贡献记录数: {total_contributions:,}")
    print(f"唯一贡献者数: {len(contributor_repos):,}")
    print(f"平均每人参与仓库数: {np.mean(list(contributor_repos.values())):.2f}")

    # 活跃贡献者
    print(f"\nTop 10 活跃贡献者 (参与仓库数):")
    for user, count in contributor_repos.most_common(10):
        print(f"  {user}: {count} 个仓库")

    # 分布统计
    repo_counts = list(contributor_repos.values())
    print(f"\n贡献者活跃度分布:")
    print(f"  中位数: {np.median(repo_counts):.0f} 个仓库")
    print(f"  参与 >= 5 个仓库: {sum(1 for c in repo_counts if c >= 5):,} 人")
    print(f"  参与 >= 10 个仓库: {sum(1 for c in repo_counts if c >= 10):,} 人")
    print(f"  参与 >= 20 个仓库: {sum(1 for c in repo_counts if c >= 20):,} 人")


def main():
    print("\n" + "=" * 60)
    print("贡献者网络合理性检查")
    print("=" * 60)

    # 分析仓库分布
    repo_sizes = analyze_repo_distribution()

    # 分析实际网络
    analyze_actual_network()

    # 检查贡献者重复性
    check_duplicate_contributors()

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

    # 给出结论
    print("\n💡 结论:")
    print("  1. 如果大多数仓库有100+贡献者，246万条边是合理的")
    print("  2. 高权重边表示核心协作团队（同时参与多个项目）")
    print("  3. 如果觉得边太多，可以考虑:")
    print("     - 过滤低权重边 (weight < 2)")
    print("     - 只保留头部活跃贡献者")
    print("     - 限制单个仓库的最大贡献者数量")


if __name__ == "__main__":
    main()