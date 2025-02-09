<h1>Crypto Twitter Analysis and Bot</h1>

<p>This project is designed to fetch, analyze, and interact with tweets related to cryptocurrency. It provides insights into market sentiment, evaluates user credibility, and allows for direct interaction via a Twitter bot.</p>

<h2>Overview</h2>

<p>The application consists of several components:</p>
<ul>
    <li><strong>Data Collection:</strong> Fetches tweets using specific cryptocurrency-related keywords.</li>
    <li><strong>Data Analysis:</strong> Analyzes tweets to extract insights such as sentiment and relevance.</li>
    <li><strong>Credibility Scoring:</strong> Evaluates the credibility of top Twitter users based on their tweet history.</li>
    <li><strong>Twitter Bot:</strong> Posts tweets and engages in discussions on selected topics or influencers.</li>
    <li><strong>Streamlit Interface:</strong> Provides a user-friendly interface for interacting with the data and bot functionalities.</li>
</ul>

<h2>Main Code Section</h2>

<p>The main code handles data fetching, analysis, and interaction with the Twitter API. It uses various libraries such as <code>requests</code>, <code>pandas</code>, and <code>streamlit</code> to process and display data.</p>

<h3>Key Functions</h3>

<ul>
    <li><code>get_tweets(query, sort, start_date, end_date)</code>
        <ul>
            <li>Fetches tweets based on the specified query and date range.</li>
            <li>Uses the Datura API to retrieve tweets with specific filters such as language and engagement metrics.</li>
            <li>Returns a list of tweets in JSON format.</li>
        </ul>
    </li>
    <li><code>get_top_tweets_df(force_refresh=False)</code>
        <ul>
            <li>Compiles top tweets over the last month using predefined cryptocurrency keywords.</li>
            <li>Checks for existing data in <code>top_tweets.csv</code> to avoid redundant API calls unless <code>force_refresh</code> is set to <code>True</code>.</li>
            <li>Returns a DataFrame of top tweets.</li>
        </ul>
    </li>
    <li><code>analyse_tweets_df(tweets_df)</code>
        <ul>
            <li>Processes the DataFrame of tweets to extract and compute various metrics.</li>
            <li>Includes user details, engagement ratios, and deduplication of tweets.</li>
            <li>Returns an enriched DataFrame with additional analysis columns.</li>
        </ul>
    </li>
    <li><code>get_top_tweeters_df(tweets_df)</code>
        <ul>
            <li>Identifies top tweeters based on the number of top tweets.</li>
            <li>Processes user data to compute statistics like follower count and tweet engagement.</li>
            <li>Returns a DataFrame of top tweeters with relevant metrics.</li>
        </ul>
    </li>
    <li><code>query_chatgpt(message)</code>
        <ul>
            <li>Utilizes OpenAI's API to analyze tweet content.</li>
            <li>Generates insights such as sentiment, relevance, and technical complexity.</li>
            <li>Returns structured analysis results for each tweet.</li>
        </ul>
    </li>
    <li><code>upload_tweet(tweet_text)</code>
        <ul>
            <li>Posts a tweet using the Twitter API.</li>
            <li>Requires authentication via OAuth and sends the tweet content as a payload.</li>
            <li>Prints the response from the Twitter API to confirm the tweet status.</li>
        </ul>
    </li>
</ul>

<h2>Credibility Score Generator</h2>

<p>This module evaluates the credibility of top Twitter users based on their cryptocurrency-related tweets. It analyzes the sentiment expressed in tweets and compares it against real coin data to generate credibility scores.</p>

<h3>Functions</h3>

<ul>
    <li><code>analyze_top_users_credibility(top_users_df, tweets_df)</code>: Analyzes credibility of top users based on their tweet history.</li>
</ul>

<h3>Workflow</h3>

<ol>
    <li>Initialize CoinGecko API for fetching coin data.</li>
    <li>Filter tweets for top users.</li>
    <li>Process each tweet to calculate credibility scores.</li>
    <li>Aggregate results and return a sorted DataFrame with user credibility scores.</li>
</ol>

<h2>Twitter Bot Control</h2>

<p>This script demonstrates how to post a tweet using the Twitter API v2. It sends a simple "Hello World!" message to Twitter.</p>

<h3>Code Explanation</h3>

<pre><code>import requests
import json

url = "https://api.twitter.com/2/tweets"

payload = json.dumps({
  "text": "Hello World! Take 3"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'OAuth oauth_consumer_key="your_consumer_key",oauth_token="your_access_token",oauth_signature_method="HMAC-SHA1",oauth_timestamp="your_timestamp",oauth_nonce="your_nonce",oauth_version="1.0",oauth_signature="your_signature"',
  'Cookie': 'guest_id=v1%3Ayour_guest_id'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
</code></pre>

<h3>Authentication</h3>

<p>The script uses OAuth 1.0a for authentication. Replace the placeholders with your actual Twitter API credentials.</p>

<h2>Usage</h2>

<ol>
    <li>Set up your environment variables for API keys and other configurations.</li>
    <li>Run the Streamlit application to interact with the data and bot functionalities.</li>
    <li>Use the provided functions to fetch and analyze tweets, evaluate user credibility, and post tweets via the bot.</li>
</ol>

<h2>Contact</h2>

<p>For questions or support, please contact <a href="mailto:yourname@example.com">yourname@example.com</a>.</p>
