from urllib.request import urlretrieve
import pandas as pd
import re
from sklearn.model_selection import train_test_split


def download_tweets(url, save_path):
    urlretrieve(url=url, filename=save_path)

# Preprocessing the data:


def clean_tweet(tweet: str) -> str:
    cleaned_tweet = tweet.lower()
    # remove "mentions" e.g. @name
    cleaned_tweet = re.sub("@\S*", "", cleaned_tweet)
    cleaned_tweet = re.sub("http\S*", "", cleaned_tweet)  # remove links
    cleaned_tweet = re.sub("www\.\S*", "", cleaned_tweet)  # remove links
    # replace repeated spaces with a single space
    cleaned_tweet = re.sub("\s+", " ", cleaned_tweet)
    cleaned_tweet = cleaned_tweet.strip()
    return cleaned_tweet


def clean_tweets(frac):
    with open("data/tweets.txt", 'r') as f:
        cleaned_tweets_list = []
        for tweet in f:
            if len(tweet.split()) >= 2:
                cleaned_tweet = clean_tweet(tweet)
                cleaned_tweets_list.append(cleaned_tweet)

    cleaned_tweets_df = pd.DataFrame({"cleaned tweets": cleaned_tweets_list})
    cleaned_tweets_df.sample(frac=frac, random_state=42).to_csv(
        "data/cleaned_tweets.csv", index=False)


def partition_into_train_test():
    cleaned_tweets_df = pd.read_csv("data/cleaned_tweets.csv")
    train_tweets_df, val_tweets_df = train_test_split(
        cleaned_tweets_df, test_size=0.2, random_state=42)
    val_tweets_df, test_tweets_df = train_test_split(
        val_tweets_df, test_size=0.5, random_state=42)

    train_tweets_df.to_csv("data/train_tweets.csv", index=False)

    splited_val_tweets_df= split_tweets(val_tweets_df)
    splited_test_tweets_df = split_tweets(test_tweets_df)

    splited_val_tweets_df.to_csv("data/val_tweets.csv", index=False)
    splited_test_tweets_df.to_csv("data/test_tweets.csv", index=False)


def split_tweets(tweets_df: pd.DataFrame):
    tweets_df.dropna(inplace=True)
    tweets_df["tokens"] = tweets_df["cleaned tweets"].apply(
        lambda x: x.split())
    
    tweets_df["lengths"] = tweets_df["tokens"].apply(len)
    tweets_df = tweets_df[tweets_df["lengths"] >= 4]

    tweets_df["begining"] = tweets_df["tokens"].apply(
        lambda x: x[:int(len(x)*0.75)])
    tweets_df["ending"] = tweets_df["tokens"].apply(
        lambda x: x[int(len(x)*0.75):])
    
    tweets_df["begining"] = tweets_df["begining"].apply(
        lambda x: " ".join(x))
    tweets_df["ending"] = tweets_df["ending"].apply(
        lambda x: " ".join(x))
    tweets_df.drop(["cleaned tweets", "tokens", "lengths"], axis=1, inplace=True)
    return tweets_df
