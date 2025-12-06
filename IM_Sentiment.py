import pandas as pd
import re
from textblob import TextBlob

df = pd.read_csv('Test.csv', engine='python')
texts = df['text']

def remove_handles(text):
    return re.sub(r'@\w+', '', text)

texts_clean = texts.apply(remove_handles)

def get_sentiment(text):
    a = TextBlob(text)
    if a.sentiment.polarity > 0:
        return 'positive'
    elif a.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

sentiments = texts_clean.apply(get_sentiment)
sentiment_df = pd.DataFrame({'sentiment': sentiments})

result_df = pd.concat([df, sentiment_df], axis=1)

positive_tweets = result_df[result_df['sentiment'] == 'positive']
negative_tweets = result_df[result_df['sentiment'] == 'negative']
neutral_tweets = result_df[result_df['sentiment'] == 'neutral']

print(positive_tweets.head())
print(negative_tweets.head())
print(neutral_tweets.head())
