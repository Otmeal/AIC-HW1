import requests
from bs4 import BeautifulSoup
from requests.models import Response
import csv
from urllib.parse import urljoin
from typing import List
import os

# 設定目標 URL 與標籤
base_url = "https://edition.cnn.com/world/"
continent_labels: List[str] = ["africa", "americas", "asia", "europe", "australia"]
# 發送請求取得首頁 HTML


def getHTML(url: str) -> Response:
    response = requests.get(url)
    if response.status_code != 200:
        print("無法取得首頁內容")
        exit()

    return response


def find_article_links(response: Response) -> List[str]:

    soup = BeautifulSoup(response.text, "html.parser")

    # 找出所有含有文章連結的 <a> 標籤（依據 class 屬性篩選）
    article_links = []
    for a_tag in soup.find_all("a", class_=lambda x: x and "container__link" in x):
        href = a_tag.get("href")
        if href and href.startswith("/"):
            full_url = urljoin(base_url, href)
            article_links.append(full_url)

    # 移除重複連結
    article_links = list(set(article_links))
    print(f"找到 {len(article_links)} 篇文章連結")
    return article_links


def fetch_article_text(article_url):
    try:
        res = requests.get(article_url)
        if res.status_code != 200:
            return None
        article_soup = BeautifulSoup(res.text, "html.parser")
        # 根據實際文章頁面，找到主要內容區域。這裡僅提供範例，需根據實際 HTML 結構調整。
        content = article_soup.find("div", class_="l-container")
        if not content:
            # 若找不到指定區塊，可以嘗試其他選擇器
            content = article_soup.find("article")
        if content:
            paragraphs = content.find_all("p")
            text = "\n".join([p.get_text(strip=True) for p in paragraphs])
            return text
        return None
    except Exception as e:
        print(f"處理 {article_url} 時發生錯誤：{e}")
        return None


# 儲存資料至 CSV 檔案
output_file = "cnn_articles_raw.csv"


if __name__ == "__main__":
    # Check if the file already exists
    file_exists = os.path.isfile(output_file)

    if file_exists:
        os.remove(output_file)
        file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["url", "text", "continent"])
            file_exists = os.path.isfile(output_file)

        for continent_label in continent_labels:
            url = urljoin(base_url, continent_label)
            response = getHTML(url)
            article_links = find_article_links(response)

            for url in article_links:
                print(f"正在處理文章：{url}")
                article_text = fetch_article_text(url)
                if article_text and len(article_text) > 100:  # 只存下有足夠文字的文章
                    writer.writerow([url, article_text, continent_label])

    print(f"資料已儲存至 {output_file}")
