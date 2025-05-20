

-- 1. agg_crypto_tweets_internal
SELECT TOP 5 * FROM agg_crypto_tweets_internal;
SELECT COUNT(*) AS total_rows FROM agg_crypto_tweets_internal;

-- 2. agg_crypto_news_internal
SELECT TOP 5 * FROM agg_crypto_news_internal;
SELECT COUNT(*) AS total_rows FROM agg_crypto_news_internal;

-- 3. agg_crypto_prices_internal
SELECT TOP 5 * FROM agg_crypto_prices_internal;
SELECT COUNT(*) AS total_rows FROM agg_crypto_prices_internal;

-- 4. agg_sentiment_over_time_internal
SELECT TOP 5 * FROM agg_sentiment_over_time_internal;
SELECT COUNT(*) AS total_rows FROM agg_sentiment_over_time_internal;

-- 5. agg_rolling_sentiment_volatility_internal
SELECT TOP 5 * FROM agg_rolling_sentiment_volatility_internal;
SELECT COUNT(*) AS total_rows FROM agg_rolling_sentiment_volatility_internal;

-- 6. agg_rsi_vs_price_internal
SELECT TOP 5 * FROM agg_rsi_vs_price_internal;
SELECT COUNT(*) AS total_rows FROM agg_rsi_vs_price_internal;

-- 7. agg_summary_metrics_internal
SELECT TOP 5 * FROM agg_summary_metrics_internal;
SELECT COUNT(*) AS total_rows FROM agg_summary_metrics_internal;


