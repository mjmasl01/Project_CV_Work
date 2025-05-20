
INSERT INTO agg_crypto_news_internal (date, source, news_count)
SELECT 
    TRY_CAST(date AS DATE),
    source,
    TRY_CAST(news_count AS BIGINT)
FROM agg_crypto_news
WHERE TRY_CAST(date AS DATE) IS NOT NULL
  AND TRY_CAST(news_count AS BIGINT) IS NOT NULL;
GO


