
INSERT INTO agg_rolling_sentiment_volatility_internal (
    date, avg_tweet_sent_roll3, avg_news_sent_roll3, avg_volatility_3d,
    avg_tweet_sent_roll7, avg_news_sent_roll7, avg_volatility_7d
)
SELECT 
    TRY_CAST(date AS DATE),
    TRY_CAST(avg_tweet_sent_roll3 AS FLOAT),
    TRY_CAST(avg_news_sent_roll3 AS FLOAT),
    TRY_CAST(avg_volatility_3d AS FLOAT),
    TRY_CAST(avg_tweet_sent_roll7 AS FLOAT),
    TRY_CAST(avg_news_sent_roll7 AS FLOAT),
    TRY_CAST(avg_volatility_7d AS FLOAT)
FROM agg_rolling_sentiment_volatility
WHERE TRY_CAST(date AS DATE) IS NOT NULL;
GO

