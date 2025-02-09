import os
# from dotenv import load_dotenv
import logging
import sys
import numpy as np
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
from tqdm.notebook import tqdm
import ast
import re
import time
from openai import AzureOpenAI
import streamlit as st

####VALUES WE NEED
API_KEY: str = st.secrets.get("API_KEY", os.getenv("API_KEY"))
MODEL: str = st.secrets.get("MODEL", os.getenv("MODEL"))
ENDPOINT: str = st.secrets.get("ENDPOINT", os.getenv("ENDPOINT"))
REGION: str = st.secrets.get("REGION", os.getenv("REGION"))
AUTH: str = st.secrets.get("AUTH", os.getenv("AUTH"))
DEPLOYMENT = MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@lru_cache(maxsize=512)
def get_tweets(query, sort, start_date, end_date):

    url = "https://apis.datura.ai/twitter"

    payload = {
        "query": query,  # "crypto"
        "sort": sort,  # "Latest"
        "start_date": start_date,  # "2025-02-06"
        "end_date": end_date,  # "2025-02-07"
        "lang": "en",
        # "verified": True,
        # "blue_verified": True,
        # "is_quote": False,
        # "is_video": False,
        # "is_image": False,
        "min_retweets": 1,
        "min_replies": 1,
        "min_likes": 1
    }
    headers = {
        "Authorization": os.getenv("TWITTER_AUTH"),
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    tweets_list = json.loads(response.text)
    return tweets_list

crypto_keyword_list = [
    "Bitcoin",
    "Ethereum",
    "Crypto",
    "Cryptocurrency",
    "DeFi",
    "Blockchain",
    "NFT",
    "Altcoins",
    "BTC",
    "ETH",
    "Stablecoins",
    "CryptoInvesting",
    "Tokenomics",
    "Shitcoin",
]

def get_top_tweets_df(force_refresh=False):
    # Get top tweets over the last month
    if os.path.exists('top_tweets.csv') and not force_refresh:
        print("Loading top tweets from csv")
        return pd.read_csv('top_tweets.csv')
    else:
        print("Compiling top tweets")
        top_tweets_list = []
        current_date = datetime.now()
        day_range = 30
        month_ago = current_date + timedelta(days=-day_range - 1)
        for n in tqdm(range(day_range)):
            start_date = month_ago + timedelta(days=n)
            # print(f'Compiling top tweets from {start_date.strftime("%Y-%m-%d")}')
            for query_string in tqdm(crypto_keyword_list):
                tweets_list = get_tweets(query=query_string, sort="Top", start_date=start_date.strftime("%Y-%m-%d"), end_date=(start_date + timedelta(days=1)).strftime("%Y-%m-%d"))
                for tweet in tweets_list:
                    top_tweets_list.append(tweet)
        top_tweets_df = pd.DataFrame(top_tweets_list)
    return top_tweets_df

def parse_json(json_str):
    if isinstance(json_str, str):
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError, TypeError):
                return {}
    return {}

def analyse_tweets_df(tweets_df):
    df = tweets_df.copy()
    df = df.astype(str)
    df['text'] = df['text'].str.replace('\n', '', regex=False)
    # print(df['user'].iloc[0])
    df.loc[:, 'user'] = df['user'].apply(lambda x: parse_json(x) if isinstance(x, str) else {})
    df.loc[:, 'user_id'] = df['user'].apply(lambda x: x.get('id', '') if isinstance(x, dict) else None)
    df.loc[:, 'follower_count'] = df['user'].apply(lambda x: int(x.get('followers_count', 0)) if isinstance(x, dict) else None)
    df = df.drop_duplicates(subset='text')
    df.loc[:, 'user_name'] = df['user'].apply(lambda x: x.get('name', '') if isinstance(x, dict) else None)
    df.loc[:, 'user_username'] = df['user'].apply(lambda x: x.get('username', '') if isinstance(x, dict) else None)
    df.loc[:, 'profile_image_url'] = df['user'].apply(lambda x: x.get('profile_image_url', '') if isinstance(x, dict) else None)
    num_columns = ['bookmark_count', 'reply_count', 'retweet_count', 'like_count', 'quote_count', 'user_id']
    df[num_columns] = df[num_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    for col in num_columns:
        df[col] = df[col].astype(int)
    df['reply_like_ratio'] = df['reply_count'] / df['like_count']
    df['bookmark_like_ratio'] = df['bookmark_count'] / df['like_count']
    df['retweet_like_ratio'] = df['retweet_count'] / df['like_count']
    return df

top_tweets_df = get_top_tweets_df()
top_tweets_df = analyse_tweets_df(top_tweets_df)
top_tweets_df.to_csv('top_tweets.csv', index=False)

def get_top_tweeters_df(tweets_df):
    # Generate top tweeters list
    unique_tweeters = tweets_df.drop_duplicates(subset='user_id', keep='first')
    user_list = unique_tweeters['user'].tolist()
    for i, item in enumerate(user_list):
        if not isinstance(item, dict):
            user_list[i] = {}
    user_df = pd.DataFrame(user_list)

    num_columns = ['id', 'favourites_count', 'followers_count', 'listed_count', 'media_count']
    user_df[num_columns] = user_df[num_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    for col in num_columns:
        user_df[col] = user_df[col].astype(int)
    tweet_counts = tweets_df['user_id'].value_counts()
    user_df['top_tweet_count'] = user_df['id'].map(tweet_counts)

    return user_df

# user_df = get_top_tweeters_df(top_tweets_df)
# user_df.to_csv('top_tweeters.csv', index=False)

client = AzureOpenAI(
    azure_endpoint=ENDPOINT, 
    api_key=API_KEY,  
    api_version="2024-05-01-preview"
)

def default_query(context, message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": message},
        ]
    )
    return response.choices[0].message.content


@lru_cache(maxsize=1024 * 8)
def query_chatgpt(message):
    system_prompt = """
    You are assisting to evaluate a variety of twitter messages.
    Summarise the intent of this message in ONLY one sentence.
    Then evaluate the relevance of this message to the topic of the crypto market as a number from 1 to 10.
    Then evaluate the technical knowledge or complexity demonstrated in this message as a number from 1 to 10.
    Then identify each cryptocurrency mentioned in the prompt. Evaluate the sentiment regarding each cryptocurrency as a number from 1 to 10, with 1 being negative and 10 being positive. List these as a python dictionary.
    Use the format below, such that the output can be read by machine. Do not include the {} in the output.
    {Sentence} -- {crypto_relevance} -- {technical_knowledge} -- {cryptocurrency_sentiment_dict}
    """
    response = client.chat.completions.create(
        model=MODEL,  # model = "deployment_name".
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
    )
    print("\tPrompt tokens:", response.usage.prompt_tokens)
    print("\tTotal tokens:", response.usage.total_tokens)
    print("\tCompletion tokens:", response.usage.completion_tokens)
    return response.choices[0].message.content

def gpt_safe_split(text, item):
    try:
        if len(text.split('--')) == 4:
            data = text.split('--')[item].strip()
            if item == 0:
                return data
            if item == 1 or item == 2:
                try:
                    return int(data)
                except ValueError:
                    return 5
            if item == 3:
                try:
                    return ast.literal_eval(data)
                except (ValueError, SyntaxError, TypeError):
                    return {}
    except Exception as e:
        print(e)
    return ['', 5, 5, {}]

# test_df = top_tweets_df.iloc[:100]

# results = []
# last_time = time.time()
# for i, row in test_df.iterrows():
#     print(f'Processing {i}')
#     response = query_chatgpt(row['text'])
#     results.append(response)
#     print(f'Completed. {time.time() - last_time} seconds.')
#     last_time = time.time()

# test_df['gpt_results'] = results

# test_df['gpt_summary'] = test_df['gpt_results'].apply(lambda x: gpt_safe_split(x, 0))
# test_df['gpt_relevance'] = test_df['gpt_results'].apply(lambda x: gpt_safe_split(x, 1))
# test_df['gpt_technical_score'] = test_df['gpt_results'].apply(lambda x: gpt_safe_split(x, 2))
# test_df['gpt_cryptocurrency_sentiment'] = test_df['gpt_results'].apply(lambda x: gpt_safe_split(x, 3))

# test_df.to_csv('test_df.csv', index=False)

def upload_tweet(tweet_text):
    url = "https://api.twitter.com/2/tweets"

    payload = json.dumps({
    "text": f"{tweet_text}"
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': AUTH,
    'Cookie': 'guest_id=v1%3A173905191714944479'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    return

top_tweets_df = pd.read_csv('top_tweets.csv')
user_df = pd.read_csv('top_tweeters.csv')

# Add Streamlit interface
def main():    
    try:
        # Sidebar for navigation
        page = st.sidebar.selectbox("Select a page", ["Home", "Daily Perspective", "Hot Tweets", "Influencers", "The Master Debater", "Direct Bot Control"])
        
        logging.info(f"Selected page: {page}")
        
        if page == "Home":
            st.write("Welcome to the OKC Twitter Agent!")
            st.write("This app fetches and analyzes tweets related to cryptocurrency.")

        elif page == "Daily Perspective":
            st.title("Daily Perspective")
            user_input = st.text_input("What viewpoint would you like to query?", "")
            if st.button("View State"):
                context = "The user would like to search twitter to find tweets related to this topic. Provide 2 keywords to search for. Make them as specific as possible. Separate them with OR. Do not respond with anything else."
                query = default_query(context, user_input)
                tweets_list = get_tweets(query, "Top", (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d"), (datetime.now() + timedelta(days=0)).strftime("%Y-%m-%d"))
                tweets_df = pd.DataFrame(tweets_list)
                print(f"Tweets fetched: {len(tweets_df)}")
                tweets_df = analyse_tweets_df(tweets_df)
                tweets_df = tweets_df.iloc[:10]

                # Classify whether each tweet is FOR or AGAINST the query.
                output_list = []
                for tweet_text in tweets_df['text']:
                    context = f"""
                    Your job is to classify whether the tweet below is FOR, NEUTRAL, or AGAINST regarding this viewpoint. 
                    Respond only with either 0 for FOR, 1 for NEUTRAL, or 2 for AGAINST.
                    The viewpoint is: {user_input}
                    """
                    output = default_query(context, tweet_text)
                    output = output.strip()
                    if output in ['0', '1', '2']:
                        output_list.append(int(output))
                    else:
                        st.warning("Invalid output received. Expected 0, 1, or 2.")
                        output_list.append(1)  # Append None or handle as needed
                tweets_df['viewpoint'] = output_list

                total_for = 0
                total_against = 0
                for i, row in tweets_df[tweets_df['viewpoint'] == 0].iterrows():
                    total_for += row['like_count'] + row['reply_count']
                for i, row in tweets_df[tweets_df['viewpoint'] == 2].iterrows():
                    total_against += row['like_count'] + row['reply_count']
                percentage = total_for/(total_for+total_against)*100

                st.markdown(f"<div style='width: 100%; background-color: red; border-radius: 5px;'>"
                             f"<div style='width: {percentage}%; background-color: green; height: 20px; border-radius: 5px;'></div>"
                             f"</div>", unsafe_allow_html=True)

                st.write(f"Overall Perspective Today: {percentage:.2f}% Positive")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"FOR")
                    st.markdown("---")  # Add a horizontal line
                    for i, row in tweets_df[tweets_df['viewpoint'] == 0].iterrows():
                        col_img, col_text = st.columns([1, 6])
                        with col_img:
                            profile_img = row['profile_image_url']
                            if profile_img:
                                st.image(profile_img, width=40)
                            else:
                                st.image("https://via.placeholder.com/40", width=40)
                        with col_text:
                            st.write(f"{row['user_name']}")
                        st.write(row['text'])
                        st.write(f"Likes: {row['like_count']},  Replies: {row['reply_count']}")

                        st.markdown("---")  # Add a horizontal line

                with col2:
                    st.write(f"AGAINST")
                    st.markdown("---")  # Add a horizontal line
                    for i, row in tweets_df[tweets_df['viewpoint'] == 2].iterrows():
                        col_img, col_text = st.columns([1, 6])
                        with col_img:
                            profile_img = row['user'].get('profile_image_url', '')
                            if profile_img:
                                st.image(profile_img, width=40)
                            else:
                                st.image("https://via.placeholder.com/40", width=40)
                        with col_text:
                            st.write(f"{row['user_name']}")
                        st.write(row['text'])
                        st.write(f"Likes: {row['like_count']},  Replies: {row['reply_count']}")

                        st.markdown("---")  # Add a horizontal line

        elif page == "Hot Tweets":
            top_tweets_df = pd.read_csv('merged_tweets_df.csv')
            st.title("Hot Tweets")
            st.write("Top Tweets from this month:")
            for i, row in top_tweets_df.iterrows():
                if i > 20:
                    break
                st.markdown("---")
                col_1, col_2, col_3 = st.columns([1, 1, 6])
                with col_1:
                    st.write(f"#{i}")
                with col_2:
                    profile_img = row['profile_image_url']
                    if profile_img:
                        st.image(profile_img, width=40)
                    else:

                        st.image("https://via.placeholder.com/40", width=40)
                with col_3:
                    st.write(row['user_name'])
                st.write(row['text'])
                st.write(f"Likes: {row['like_count']},  Replies: {row['reply_count']}")
                st.write(f"Crypto Relevance: {row['crypto_relevance']}")
                st.write(f"Technical Knowledge: {row['technical_knowledge']}")
                st.write(f"Cryptocurrency Sentiments: {row['crypto_sentiments']}")
                    



        elif page == "Influencers":
            user_df = pd.read_csv('top_tweeters.csv')
            st.title("Influencers")
            st.write("Top Influencers from this month:")
            user_df = user_df.sort_values(by='influence_metric', ascending=False)
            user_df.reset_index(drop=True, inplace=True)
            for j, row in user_df.iterrows():
                st.markdown("---")
                col_1, col_2, col_3 = st.columns([1, 1, 6])

                with col_1:
                    st.write(f"#{j}")
                with col_2:
                    profile_img = row['profile_image_url']
                    if profile_img:
                        st.image(profile_img, width=40)
                    else:
                        st.image("https://via.placeholder.com/40", width=40)
                with col_3:
                    st.write(row['name'])
                    st.write(f"Followers: {row['followers_count']}, Top Tweets: {row['top_tweet_count']}, Influence: {row['influence_metric']}")
                    

                
            
        elif page == "The Master Debater":
            st.title("The Master Debater")
            st.write("Set a bot up to weigh in on a crypto topic")
            query = st.selectbox("Choose a target category", ["Topic", "Influencer"])
            if query == "Topic":
                viewpoint = st.text_input("Enter a viewpoint")
                style = st.text_input("Enter a style")
                spiciness = st.slider("Select Spiciness Level", min_value=1, max_value=10, value=5, step=1)
                if st.button("Generate Tweet"):
                    context = f"""
                    Create a tweet that is in the style '{style}' about the topic provided, with a spiciness level of {spiciness}.
                    The shorter the better. You must keep it to less than 4 sentences. Include only the text of the tweet.
                    """
                    new_tweet = default_query(context, viewpoint)
                    st.write("Proposed Tweet:")
                    st.write(f"{new_tweet}")
                    if st.button("Post Tweet"):
                        upload_tweet(new_tweet)
            elif query == "Influencer":
                user_df = pd.read_csv('top_tweeters.csv')

                influencer = st.selectbox("Choose an influencer", user_df['user_name'].sort_values(by='top_tweet_count', ascending=False).tolist())
                inf_tweet = st.selectbox("Choose a tweet", user_df[user_df['user_name'] == influencer]['text'].tolist())
                viewpoint = st.text_input("Enter a viewpoint")
                style = st.text_input("Enter a style")
                spiciness = st.slider("Select Spiciness Level", min_value=1, max_value=10, value=5, step=1)
                if st.button("Generate Tweet"):
                    context = f"""
                    Create a tweet that is in the style '{style}', with a spiciness level of {spiciness}.
                    This tweet must be in reply to {influencer}, who recently tweeted the text provided.
                    The shorter the better. You must keep it to less than 4 sentences. Include only the text of the tweet.
                    """
                    new_tweet = default_query(context, viewpoint)
                    st.write("Proposed Tweet:")
                    st.write(f"{new_tweet}")
                    if st.button("Post Tweet"):
                        upload_tweet(new_tweet)

        elif page == "Direct Bot Control":
            st.title("Direct Bot Control")
            st.write("Send a tweet directly via the bot")
            tweet_text = st.text_input("Enter a tweet")
            if st.button("Send Tweet"):
                upload_tweet(tweet_text)

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        # Print full traceback
        import traceback

        logging.error(traceback.format_exc())

# Run the Streamlit app
if __name__ == "__main__":
    try:
        logging.info("Starting application")
        main()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        logging.error(traceback.format_exc())
