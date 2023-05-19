import csv
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from collections import defaultdict
from datetime import datetime
import pandas as pd



def check_words_in_text(words, text):
    words_present = [word.lower() in text.lower() for word in words]
    return all(words_present)

   
def extract_text(url):
    try:
        response = requests.get(url, timeout=5)  # Set a timeout value for the request
        response.raise_for_status()  # Raise an exception if the request was not successful
        soup = BeautifulSoup(response.text, 'html.parser')
        p_tags = soup.find_all('p')
        text = ' '.join(tag.get_text() for tag in p_tags)
        return text
    except (requests.exceptions.RequestException, ValueError) as e:
        #print(f"An error occurred: {e}")
        return "An error occurred"

def read_from_csv(csv_file):
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            {row['link']: row for row in reader}
            existing_data = {row['link']: row for row in reader}
    except FileNotFoundError:
        existing_data = {}
    return existing_data

def write_to_csv(csv_file, existing_data):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['date', 'title', 'link', 'sentiment', 'sentiment_type','relevant']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in existing_data.values():
            data['sentiment'] = round(float(data['sentiment']), 2)
            writer.writerow(data)


def sentiment_analysis(item, existing_data, articles_by_date,Query):

    published_date = datetime.strptime(item.pubDate.text, '%a, %d %b %Y %H:%M:%S %Z').date()
    full_text = extract_text(item.link.text)
  
    if(Query== "Business" or Query == "India"  or Query== "World"):
        relevant= True
    else:
        query_words = Query.split()  # split your query into individual words
        titleisrelevant = check_words_in_text(query_words,item.title.text)
        newsisrelevant = check_words_in_text(query_words,full_text)
        if(titleisrelevant== True or newsisrelevant == True):
              relevant= True
        else:
            relevant= False
    
    #full_text = item.description.text
    analysis = TextBlob(full_text)
    sentiment = analysis.sentiment.polarity
    sentiment_type = classify_sentiment(sentiment)

    articles_by_date[published_date].append({
        'title': item.title.text,
        'description': item.description.text,
        'link': item.link.text,
        'sentiment_type': sentiment_type,
        'sentiment': sentiment,
        'relevant':relevant
    })

    existing_data[item.link.text] = {
        'date': published_date.strftime('%Y-%m-%d'),
        'title': item.title.text,
        'link': item.link.text,
        'sentiment': sentiment,
        'sentiment_type': sentiment_type,
        'relevant':relevant
    }

def classify_sentiment(sentiment):
    if sentiment > 0.5:
        return "Extremely Positive"
    elif sentiment > 0:
        return "Positive"
    elif sentiment < -0.5:
        return "Extremely Negative"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"


def ExtractSentiment(ticker:'Business',Query:"Business",pagesize:4):

    BusinesstopicID = "CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JXVnVMVWRDR2dKSlRpZ0FQAQ"
    IndiatopicID = "CAAqJQgKIh9DQkFTRVFvSUwyMHZNRE55YXpBU0JXVnVMVWRDS0FBUAE"
    WorldtopicID = "CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JXVnVMVWRDR2dKSlRpZ0FQAQ"

    topicID=""
    
    if(ticker=="Business"):
        topicID= BusinesstopicID
    elif(ticker=="India"):
        topicID= IndiatopicID
    elif(ticker=="World"):
        topicID= WorldtopicID

    if(ticker=="Business" or ticker=="India" or ticker=="World"):
        rss_url =f"https://news.google.com/rss/topics/{topicID}?hl=en-IN&gl=IN&ceid=IN:en"
    else:
        rss_url = f"https://news.google.com/rss/search?q={Query}&hl=en-IN&gl=IN&ceid=IN:en"
    

    csv_file =  'News/'+ticker+'_news_sentiment.csv'

    response = requests.get(rss_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')
        articles_by_date = defaultdict(list)
        existing_data = read_from_csv(csv_file)
        overall_sentiments = {}

        for item in soup.find_all('item')[:pagesize]:
            link = item.link.text
            if link in existing_data:
                sentiment = float(existing_data[link]['sentiment'])
                sentiment_type = existing_data[link]['sentiment_type']
                relevant = existing_data[link]['relevant']
                published_date = datetime.strptime(existing_data[link]['date'], '%Y-%m-%d').date()
                # Add existing data to articles_by_date
                articles_by_date[published_date].append({
                    'title': existing_data[link]['title'],
                    'description': item.description.text,
                    'link': link,
                    'sentiment_type': sentiment_type,
                    'sentiment': sentiment,
                    'relevant':relevant
                })
            else:  
                sentiment_analysis(item, existing_data, articles_by_date,Query)

        # Calculate overall sentiment for each date
        for date, articles in articles_by_date.items():
            overall_sentiment = sum(article['sentiment'] for article in articles) / len(articles)
            overall_sentiments[date] = overall_sentiment

        # Write new data to the CSV
        write_to_csv(csv_file, existing_data)
        df = pd.DataFrame.from_dict(overall_sentiments, orient='index', columns=['sentiment'])
        df.index.name = 'Date'
        return df
    else:
        print('Error fetching news')

def GetSentimentScore(ticker):

    csv_file =  'News/'+ticker+'_news_sentiment.csv'
    df = pd.DataFrame(columns=['date','title', 'sentiment'])
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            df = pd.DataFrame(list(reader))
            #df = df[df['relevant'] == 'True']
    except FileNotFoundError:
        print("Error")
    return  df[['date','title','sentiment']]