from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime, timedelta
import time

def analyze_top_users_credibility(top_users_df, tweets_df):
    """
    Analyze credibility of top users based on their tweet history.
    
    Parameters:
    top_users_df (pandas.DataFrame): DataFrame with column:
        - username: usernames of top 10 users
    tweets_df (pandas.DataFrame): DataFrame with columns:
        - username: creator's username
        - date: date of tweet
        - sentiment: dictionary of coin sentiments (0-10)
    
    Returns:
    pandas.DataFrame: Analysis of top users' credibility scores
    """
    # Initialize CoinGecko API
    cg = CoinGeckoAPI()
    
    def is_tweet_recent(date_str):
        """Check if tweet is within last 10 days"""
        tweet_date = datetime.strptime(date_str, '%Y-%m-%d')
        current_date = datetime.now()
        return (current_date - tweet_date).days <= 10
    
    def get_price_change(coin, date_str):
        """Get price change over 10 days for a given coin"""
        try:
            # Skip if tweet is too recent
            if is_tweet_recent(date_str):
                return None
                
            start_date = datetime.strptime(date_str, '%Y-%m-%d')
            end_date = start_date + timedelta(days=10)
            
            from_timestamp = int(start_date.timestamp())
            to_timestamp = int(end_date.timestamp())
            
            price_data = cg.get_coin_market_chart_range_by_id(
                coin,
                vs_currency='usd',
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            
            if len(price_data['prices']) >= 2:
                initial_price = price_data['prices'][0][1]
                final_price = price_data['prices'][-1][1]
                return ((final_price - initial_price) / initial_price) * 100
            
            return None
            
        except Exception as e:
            print(f"Error getting price data for {coin} on {date_str}: {str(e)}")
            return None
    
    def calculate_credibility(sentiment, price_change):
        """Calculate credibility score between -1 and 1"""
        if price_change is None:
            return None
            
        # Convert sentiment from 0-10 scale to -1 to 1 scale
        normalized_sentiment = (sentiment - 5) / 5
        normalized_price_change = max(min(price_change / 50, 1), -1)
        return normalized_sentiment * normalized_price_change
    
    def process_tweet(row):
        """Process a single tweet and return credibility scores"""
        credibility_dict = {}
        
        # Skip processing if tweet is too recent
        if is_tweet_recent(row['date']):
            print(f"Skipping recent tweet from {row['date']}")
            return {}
            
        for coin, sentiment in row['sentiment'].items():
            time.sleep(1.2)  # Respect API rate limits
            price_change = get_price_change(coin, row['date'])
            credibility = calculate_credibility(sentiment, price_change)
            credibility_dict[coin] = credibility
            
        return credibility_dict
    
    # Filter tweets for top users
    top_usernames = set(top_users_df['username'])
    relevant_tweets = tweets_df[tweets_df['username'].isin(top_usernames)].copy()
    
    # Process each tweet
    relevant_tweets['credibility'] = relevant_tweets.apply(process_tweet, axis=1)
    
    # Calculate user statistics
    user_stats = []
    
    for username in top_usernames:
        user_tweets = relevant_tweets[relevant_tweets['username'] == username]
        
        # Get all credibility scores for this user
        all_scores = []
        total_predictions = 0
        recent_predictions = 0
        
        for cred_dict in user_tweets['credibility']:
            total_predictions += len(user_tweets['sentiment'].iloc[0])
            if cred_dict:  # If not empty (not a recent tweet)
                valid_scores = [score for score in cred_dict.values() if score is not None]
                if valid_scores:
                    all_scores.extend(valid_scores)
            else:
                recent_predictions += len(user_tweets['sentiment'].iloc[0])
        
        if all_scores:
            stats = {
                'username': username,
                'average_credibility': round(sum(all_scores) / len(all_scores), 3),
                'num_tweets': len(user_tweets),
                'predictions_analyzed': len(all_scores),
                'recent_predictions': recent_predictions,
                'best_prediction': round(max(all_scores), 3) if all_scores else None,
                'worst_prediction': round(min(all_scores), 3) if all_scores else None,
                'prediction_stddev': round(pd.Series(all_scores).std(), 3) if len(all_scores) > 1 else None
            }
            user_stats.append(stats)
    
    # Create and sort results DataFrame
    results_df = pd.DataFrame(user_stats)
    results_df = results_df.sort_values('average_credibility', ascending=False).reset_index(drop=True)
    
    return results_df

# Example usage:


# Sample top users DataFrame
top_users = pd.DataFrame({
    'username': ['crypto_expert', 'trader_jane', 'bitcoin_guru']
})

# Sample tweets DataFrame
tweets = pd.DataFrame({
    'username': ['crypto_expert', 'trader_jane', 'bitcoin_guru', 'other_user'],
    'date': ['2024-11-11', '2024-11-11', '2024-11-11', '2024-11-11'],
    'sentiment': [
        {'bitcoin': 8, 'ethereum': 7},
        {'bitcoin': 3, 'ethereum': 4},
        {'bitcoin': 9, 'dogecoin': 8},
        {'ethereum': 5, 'dogecoin': 6}
    ]
})

'''
# Would be used to load real data from our API Analytics.
top_users = pd.read_csv('top_tweeters.csv')
tweets = pd.read_csv('top_tweets.csv')
'''

# Run analysis
results = analyze_top_users_credibility(top_users, tweets)
print(results)
