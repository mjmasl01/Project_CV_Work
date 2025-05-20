
CREATE TABLE agg_crypto_tweets_internal (
    date DATE,
    tweet_count BIGINT
);
GO


CREATE TABLE agg_crypto_news_internal (
    date DATE,
    source NVARCHAR(255),
    news_count BIGINT
);
GO


CREATE TABLE agg_crypto_prices_internal (
    date DATE,
    avg_open FLOAT,
    avg_close FLOAT,
    avg_high FLOAT,
    avg_low FLOAT,
    avg_volume FLOAT
);
GO


CREATE TABLE agg_sentiment_over_time_internal (
    date DATE,
    avg_tweet_sentiment FLOAT,
    avg_news_sentiment FLOAT
);
GO


CREATE TABLE agg_rolling_sentiment_volatility_internal (
    date DATE,
    avg_tweet_sent_roll3 FLOAT,
    avg_news_sent_roll3 FLOAT,
    avg_volatility_3d FLOAT,
    avg_tweet_sent_roll7 FLOAT,
    avg_news_sent_roll7 FLOAT,
    avg_volatility_7d FLOAT
);
GO


CREATE TABLE agg_rsi_vs_price_internal (
    date DATE,
    rsi_14 FLOAT,
    [Close] FLOAT
);
GO


CREATE TABLE agg_summary_metrics_internal (
    date DATE,
    avg_close FLOAT,
    avg_return_1d FLOAT,
    avg_tweet_sentiment FLOAT,
    avg_news_sentiment FLOAT,
    volatility_3d FLOAT,
    avg_rsi_14 FLOAT
);
GO

