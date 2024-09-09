# bbc-news-analysis
The news dataset is crwaled from [BBC](https://www.bbc.com/) with keyword Microsoft. The inital csv file is bbc_news_with_title.csv

For traning different models, we label information manually as different ways. 

The LogesticRegression and XGBOOST model are trained by bbc_news_with_title-labels.csv dataset

The Bert model is trained and tested by CoNLL2003 format dataset.

In this stage, we decide to collect the titles about Microsost Corp of BBC news website. The collection process stated as following steps:
1. Access directly to BBC website and using keyword "Microsoft" in search bar.
2. Collect the Url of more than 100 news manually and store it in a csv file (i.e. "bbc_news.csv").
3. Use pandas library to read the csv file just stored.
4. Use requests library and BeautifulSoup to download news content in HTML format.
5. Use json library to access through multiple class of json structure (contents of the news are in json format) to get the title.
6. Store all titles in a new csv file (i.e. "bbc_news_with_title.csv").

Special thanks to Yixuan Li for support me in this project
