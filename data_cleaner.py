import pandas as pd
import re
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 若尚未下載所需資源，請執行以下指令（只需執行一次）
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# 讀取 CSV 檔案，假設檔案名稱為 "cnn_africa_articles.csv"
input_file = "cnn_articles_raw.csv"
df = pd.read_csv(input_file)

# 定義停用字與詞形還原工具
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # 移除 HTML 標籤
    text = BeautifulSoup(text, "html.parser").get_text()
    # 轉小寫
    text = text.lower()
    # 移除標點符號
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 移除數字
    text = re.sub(r"\d+", "", text)
    # 斷詞
    tokens = nltk.word_tokenize(text)
    # 移除停用字，並進行詞形還原
    tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    # 合併回字串
    return " ".join(tokens)


# 對 'text' 欄位進行清理
df["text"] = df["text"].apply(clean_text)

# 將清理後的結果寫入新的 CSV 檔案
output_file = "cnn_articles_cleaned.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"清理後的資料已儲存至 {output_file}")
