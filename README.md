# AIC-HW1

## Dataset Documentation

The dataset contains news articles and the continents they belong to. It currently includes only the news displayed on [CNN World Edition](https://edition.cnn.com/world) as of March 9, 2025.

In the GitHub repository, you will find several Python scripts. The `crawler.py` script is used to scrape news from the CNN website. You can modify the crawler to collect news from other websites as well.

The amount of data from each continent:

| Continent | number of entrys |
| --------- | ---------------- |
| Africa    | 47               |
| Asia      | 44               |
| Europe    | 33               |
| Australia | 20               |
| Americas  | 17               |
| Total     | 161              |

### Details of `crawler.py`

The Python script is a web crawler designed to scrape news articles from CNN's World section and categorize them by continent. Below are the details of its process:

- The list of continent labels is predefined as:`[africa, americas, asia, europe, australia]`.
- The crawler iterates through `base_url`(CNN World Edition) + label.
- It scrapes news articles under each continent URL and labels them accordingly.
- News articles with fewer than 100 characters are discarded.

### Clean up script

The script `data_cleaner.py is` responsible for preprocessing and cleaning the news dataset to enhance its quality before further analysis or modeling. It performs the following key steps:

- Remove HTML tags using BeautifulSoup to extract only plain text.
- Convert text to lowercase to maintain uniformity.
- Remove punctuation and special characters to reduce noise.
- Remove numbers to focus only on textual content.
- Tokenize text using NLTK (nltk.word_tokenize).
- Remove stopwords (common words like "the", "is", "and") to keep only meaningful words.
- Lemmatize words (convert words to their base form, e.g., "running" → "run").

This script ensures the dataset is structured, noise-free, and ready for classification or clustering tasks.

### Oversampling script

The original collected dataset is imbalanced. `oversample.py` addresses this issue by performing oversampling, ensuring that all continent categories have an equal number of samples. There are some detail of it:

- It identifies the largest category in terms of article count.
- For smaller categories, it randomly resamples articles with replacement (replace=True) until all categories have the same number of samples.
- The oversampled data is shuffled to prevent order bias.
