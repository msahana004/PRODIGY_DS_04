------------#Step 1: Import Libraries--------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob

-------------#Step 2: Load Dataset-------------
df = pd.read_csv(r"C:\Users\admin\Documents\Prodigy Infotech\Task 4\Dataset\Tweets.csv")
df.head()

-------------#Step 3: Understand Data-----------
#Dataset Info
df.info()

#Check Missing Values
df.isnull().sum()

-------------#Step 4.1: Drop unnecessary columns-------------
df = df.drop(columns=[
    'negativereason', 
    'negativereason_confidence', 
    'airline_sentiment_gold',
    'negativereason_gold', 
    'tweet_coord', 
    'tweet_location', 
    'user_timezone'
])

#Step 4.2: Confirm missing values
df.isnull().sum()

-------------#Step 5: Select Text Column-------------
df[['text', 'airline_sentiment']].head()

-------------#Step 6: Sentiment Visualization---------------
#Count sentiment distribution
sns.countplot(x='airline_sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

-------------#Step 7: Sentiment by Airline----------------
sns.countplot(x='airline', hue='airline_sentiment', data=df)
plt.xticks(rotation=45)
plt.title("Sentiment by Airline")
plt.show()

------------#Step 8: Text Length Analysis---------------
df['text_length'] = df['text'].apply(len)

sns.histplot(df['text_length'], bins=30)
plt.title("Tweet Length Distribution")
plt.show()

-----------#Step 9: Polarity Analysis using TextBlob------------
# Polarity: -1 (negative) to 1 (positive)
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

plt.figure(figsize=(8,5))
sns.histplot(df['polarity'], bins=30, kde=True, color='lightgreen')
plt.title("Sentiment Polarity Distribution")
plt.xlabel("Polarity Score")
plt.ylabel("Number of Tweets")
plt.show()
