
INSERT INTO agg_crypto_tweets_internal (date, tweet_count)
SELECT 
    TRY_CAST(date AS DATE),
    TRY_CAST(tweet_count AS BIGINT)
FROM agg_crypto_tweets
WHERE TRY_CAST(date AS DATE) IS NOT NULL
  AND TRY_CAST(tweet_count AS BIGINT) IS NOT NULL;
GO



