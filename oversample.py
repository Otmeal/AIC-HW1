import pandas as pd

# 讀取已清理的 CSV 檔案
input_file = "cnn_articles_cleaned.csv"
df = pd.read_csv(input_file)

# 顯示各類別原始分布
print("原始資料各類別分布：")
print(df["continent"].value_counts())

# 找出各類別樣本數的最大值
max_count = df["continent"].value_counts().max()


# 對每個類別進行 oversampling（有放回抽樣）使其樣本數達到最大值
df_list = []
for label, group in df.groupby("continent"):
    if len(group) < max_count:
        group_oversampled = group.sample(n=max_count, replace=True, random_state=42)
    else:
        group_oversampled = group
    df_list.append(group_oversampled)

# 合併 oversampling 後的各類資料，並打亂順序
df_oversampled = pd.concat(df_list)
df_oversampled = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

# 顯示 oversampling 後各類別分布
print("\nOversampling 後各類別分布：")
print(df_oversampled["continent"].value_counts())

# 將 oversampling 後的結果寫入新的 CSV 檔案
output_file = "cnn_articles_oversampled.csv"
df_oversampled.to_csv(output_file, index=False, encoding="utf-8")
print(f"\n平衡後的資料已儲存至 {output_file}")
