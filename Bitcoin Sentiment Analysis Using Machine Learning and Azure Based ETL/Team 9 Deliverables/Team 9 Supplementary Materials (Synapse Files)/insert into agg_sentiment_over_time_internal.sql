
INSERT INTO agg_sentiment_over_time_internal (date, avg_tweet_sentiment, avg_news_sentiment)
SELECT 
    TRY_CAST(date AS DATE),
    TRY_CAST(avg_tweet_sentiment AS FLOAT),
    TRY_CAST(avg_news_sentiment AS FLOAT)
FROM agg_sentiment_over_time
WHERE TRY_CAST(date AS DATE) IS NOT NULL;
GO


