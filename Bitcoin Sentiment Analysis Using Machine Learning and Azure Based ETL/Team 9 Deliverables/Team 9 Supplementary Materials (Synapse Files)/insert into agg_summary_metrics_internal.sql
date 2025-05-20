
INSERT INTO agg_summary_metrics_internal (
    date, avg_close, avg_return_1d,
    avg_tweet_sentiment, avg_news_sentiment,
    volatility_3d, avg_rsi_14
)
SELECT 
    TRY_CAST(date AS DATE),
    TRY_CAST(avg_close AS FLOAT),
    TRY_CAST(avg_return_1d AS FLOAT),
    TRY_CAST(avg_tweet_sentiment AS FLOAT),
    TRY_CAST(avg_news_sentiment AS FLOAT),
    TRY_CAST(volatility_3d AS FLOAT),
    TRY_CAST(avg_rsi_14 AS FLOAT)
FROM agg_summary_metrics
WHERE TRY_CAST(date AS DATE) IS NOT NULL;
GO


