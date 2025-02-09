import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

####VALUES WE NEED
API_KEY: str = os.getenv("API_KEY")
MODEL: str = os.getenv("MODEL")
ENDPOINT: str = os.getenv("ENDPOINT")
REGION: str = os.getenv("REGION")
DEPLOYMENT = MODEL

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
import streamlit as st

from openai import AzureOpenAI

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
    num_columns = ['bookmark_count', 'reply_count', 'retweet_count', 'like_count', 'quote_count', 'user_id']
    df[num_columns] = df[num_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    for col in num_columns:
        df[col] = df[col].astype(int)
    df['reply_like_ratio'] = df['reply_count'] / df['like_count']
    df['bookmark_like_ratio'] = df['bookmark_count'] / df['like_count']
    df['retweet_like_ratio'] = df['retweet_count'] / df['like_count']
    df['like_follower_ratio'] = df['like_count'] / df['follower_count']
    return df

# top_tweets_df = get_top_tweets_df()
# top_tweets_df = analyse_tweets_df(top_tweets_df)
# top_tweets_df.to_csv('top_tweets.csv', index=False)

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

# def create_tweet_image(tweet_dict):
#     # Load the user's icon
#     user_icon_url = tweet_dict.get('user', {}).get('profile_image_url', '')
#     response = requests.get(user_icon_url)
#     user_icon = Image.open(BytesIO(response.content)).resize((50, 50))


#     # Create a new image with rounded edges
#     width, height = 400, 200
#     image = Image.new('RGB', (width, height), color='white')
#     draw = ImageDraw.Draw(image)

#     # Draw rounded rectangle
#     radius = 20
#     draw.rounded_rectangle([0, 0, width, height], radius=radius, fill='lightgray')

#     # Draw user icon
#     image.paste(user_icon, (10, 10))

#     # Draw tweet body
#     tweet_body = tweet_dict.get('text', 'No content')
#     font = ImageFont.load_default()
#     draw.text((70, 10), tweet_body, fill='black', font=font)


#     # Draw likes and replies
#     likes = tweet_dict.get('like_count', 0)
#     replies = tweet_dict.get('reply_count', 0)
#     draw.text((10, 160), f"Likes: {likes}  Replies: {replies}", fill='black', font=font)


#     # Save or display the image
#     image_path = f'tweet_{tweet_dict.get("user_name", "unknown")}.png'
#     image.save(image_path)
#     return image_path

# # Example usage
# tweet_example = {
#     'text': 'This is an example tweet!',
#     'user': {'profile_image_url': 'https://example.com/user_icon.png'},
#     'like_count': 10,
#     'reply_count': 5
# }

# image_path = create_tweet_image(tweet_example)


# Add Streamlit interface
def main():    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select a page", ["Home", "Daily Perspective", "Hot Tweets", "Influencers", "The Master Debater"])
    
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
            # print(tweets_df.columns)
            # print(tweets_df.iloc[0])
            tweets_df = analyse_tweets_df(tweets_df)
            tweets_df = tweets_df.iloc[:10]
            print(f"Tweets fetched: {len(tweets_df)}")
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
                        print(row['user'])
                        profile_img = row['user'].get('profile_image_url', '')
                        print(profile_img)
                        if profile_img:
                            st.image(profile_img, width=40)
                        else:
                            st.image("https://via.placeholder.com/40", width=40)
                    with col_text:
                        st.write(f"{row['user_name']}")
                        st.write(f"{row['user_username']}")
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
        st.title("Hot Tweets")
        st.write("Enter a query to fetch tweets related to cryptocurrency.")
        query = st.text_input("Query", "Bitcoin")
        if st.button("Fetch Tweets"):
            st.write("Top Tweets:")
            st.dataframe(top_tweets_df)
    elif page == "Influencers":
        st.title("Influencers")
        st.write("Enter a query to fetch influencers related to cryptocurrency.")
        query = st.text_input("Query", "Bitcoin")
        if st.button("Fetch Influencers"):
            st.write("Top Influencers:")
            st.dataframe(user_df)
    elif page == "The Master Debater":
        st.title("The Master Debater")
        st.write("Set a bot up to weigh in on a crypto topic")
        query = st.selectbox("Choose a target category", ["Topic", "Influencer"])
        if query == "Topic":
            topic = st.text_input("Enter a topic")
            if st.button("Activate Debate Bot"):
                st.write("Debate bot activated successfully!")
                st.write("The bot will now send 10 tweets about this Topic over the next 24 hours..")
        elif query == "Influencer":
            influencer = st.text_input("Enter an influencer")
            if st.button("Activate Debate Bot"):
                st.write("Debate bot activated successfully!")
                st.write("The bot will now send 10 tweets about this Influencer over the next 24 hours..")

# Run the Streamlit app
if __name__ == "__main__":

    main()
