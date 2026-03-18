from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class NextTokenDataset(Dataset):
    def __init__(self, tokenized_tweets) -> None:
        super().__init__()
        self.X = []
        self.y = []
        for tweet in tokenized_tweets:
            tweet_length = len(tweet)
            if tweet_length >= 4:
                X_item = []
                y_item = []
                for i in range(2,tweet_length):
                    X_item.append(tweet[:i])
                    y_item.append(tweet[i])
                self.X.append(X_item)
                self.y.append(y_item)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def create_next_token_dataset_from_csv(path_to_csv, tokenizer):
    tweets_df = pd.read_csv(path_to_csv)
    tweets_df.dropna(inplace=True)

    tokenized_tweets = tokenizer(tweets_df["cleaned tweets"].to_list())["input_ids"]
    return NextTokenDataset(tokenized_tweets)

def get_next_token_collate_fn(tokenizer):
    def next_token_collate_fn(batch):
        tweet_groups, target_groups =  zip(*batch)

        total_tweets = []
        for group in tweet_groups:
            total_tweets.extend(group)

        total_targets = []
        for group in target_groups:
            total_targets.extend(group)

        tweets_tensors = [torch.tensor(tweet) for tweet in total_tweets]

        targets = torch.tensor(total_targets)
        lengths = torch.tensor([len(tweet) for tweet in total_tweets], dtype =torch.int64)
        x_padded = pad_sequence(tweets_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
        return x_padded, targets, lengths

    return next_token_collate_fn

