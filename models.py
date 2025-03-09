import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import adjusted_rand_score, make_scorer, precision_score
from sklearn.cluster import KMeans

# 1. 讀取已清理且(可)經過 oversampling 的 CSV 資料
data_file = "cnn_articles_cleaned.csv"
df = pd.read_csv(data_file)

# 檢查必要欄位是否存在
if "text" not in df.columns or "continent" not in df.columns:
    raise ValueError("CSV 資料中必須包含 'text' 與 'continent' 欄位")

# 2. 建立 TF-IDF 特徵
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["continent"]

# -------------------------------
# (A) 進行 k-Fold Cross-Validation 來評估 Naive Bayes 和 SVM

print("=== 交叉驗證 (Cross-Validation) 評估 ===")

# 定義想要的多種評估指標 (macro 代表每個類別的分數平均)
scoring = {
    "acc": "accuracy",
    "prec": make_scorer(precision_score, average="macro", zero_division=1),
    "rec": "recall_macro",
    "f1": "f1_macro",
}

# === Naive Bayes 交叉驗證 ===
nb_model = MultinomialNB()
nb_cv_results = cross_validate(nb_model, X, y, cv=5, scoring=scoring, n_jobs=-1)

# 計算並印出各指標平均值與標準差
print("\n[Naive Bayes - Cross Validation]")
print(
    f"Accuracy: {nb_cv_results['test_acc'].mean():.3f} ± {nb_cv_results['test_acc'].std():.3f}"
)
print(
    f"Precision (macro): {nb_cv_results['test_prec'].mean():.3f} ± {nb_cv_results['test_prec'].std():.3f}"
)
print(
    f"Recall (macro): {nb_cv_results['test_rec'].mean():.3f} ± {nb_cv_results['test_rec'].std():.3f}"
)
print(
    f"F1-score (macro): {nb_cv_results['test_f1'].mean():.3f} ± {nb_cv_results['test_f1'].std():.3f}"
)

# === SVM 交叉驗證 ===
svm_model = LinearSVC(max_iter=10000)
svm_cv_results = cross_validate(svm_model, X, y, cv=5, scoring=scoring, n_jobs=-1)

print("\n[SVM - Cross Validation]")
print(
    f"Accuracy: {svm_cv_results['test_acc'].mean():.3f} ± {svm_cv_results['test_acc'].std():.3f}"
)
print(
    f"Precision (macro): {svm_cv_results['test_prec'].mean():.3f} ± {svm_cv_results['test_prec'].std():.3f}"
)
print(
    f"Recall (macro): {svm_cv_results['test_rec'].mean():.3f} ± {svm_cv_results['test_rec'].std():.3f}"
)
print(
    f"F1-score (macro): {svm_cv_results['test_f1'].mean():.3f} ± {svm_cv_results['test_f1'].std():.3f}"
)


# -------------------------------
# (C) k-Means 聚類 (Unsupervised)
num_clusters = len(df["continent"].unique())
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_

ari = adjusted_rand_score(y, cluster_labels)
print("\n=== K-Means Clustering ===")
print("Adjusted Rand Index (ARI):", ari)
