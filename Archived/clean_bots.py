import pandas as pd

df = pd.read_csv("../data/contributors_withContributionCount.csv")

# bots.txt，每行一个名字
with open("../data/bots.txt", "r", encoding="utf-8") as f:
    bot_list = [line.strip() for line in f if line.strip()]
df_cleaned = df[~df["contributor"].isin(bot_list)]

df_cleaned.to_csv("cleaned_file.csv", index=False)

print(f"清洗完成！原始数据 {len(df)} 行，删除机器人 {len(df) - len(df_cleaned)} 行，剩下 {len(df_cleaned)} 行。")