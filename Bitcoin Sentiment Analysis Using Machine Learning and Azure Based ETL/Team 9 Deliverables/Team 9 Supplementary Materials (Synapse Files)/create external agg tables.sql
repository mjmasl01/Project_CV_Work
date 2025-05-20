

-- Aggregated Tweet Count
CREATE EXTERNAL TABLE agg_crypto_tweets (
    date DATE,
    tweet_count BIGINT
)
WITH (
    LOCATION = 'agg_crypto_tweets',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO

-- Aggregated News Count
CREATE EXTERNAL TABLE agg_crypto_news (
    date DATE,
    source NVARCHAR(255),
    news_count BIGINT
)
WITH (
    LOCATION = 'agg_crypto_news_by_source',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO


-- Daily Price Averages
CREATE EXTERNAL TABLE agg_crypto_prices (
    date DATE,
    avg_open FLOAT,
    avg_close FLOAT,
    avg_high FLOAT,
    avg_low FLOAT,
    avg_volume FLOAT
)
WITH (
    LOCATION = 'agg_crypto_prices',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO


-- Sentiment Over Time
CREATE EXTERNAL TABLE agg_sentiment_over_time (
    date DATE,
    avg_tweet_sentiment FLOAT,
    avg_news_sentiment FLOAT
)
WITH (
    LOCATION = 'agg_sentiment_over_time',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO


-- Rolling Sentiment & Volatility
CREATE EXTERNAL TABLE agg_rolling_sentiment_volatility (
    date DATE,
    avg_tweet_sent_roll3 FLOAT,
    avg_news_sent_roll3 FLOAT,
    avg_volatility_3d FLOAT,
    avg_tweet_sent_roll7 FLOAT,
    avg_news_sent_roll7 FLOAT,
    avg_volatility_7d FLOAT
)
WITH (
    LOCATION = 'agg_rolling_sentiment_volatility',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO


-- RSI vs Close Price
CREATE EXTERNAL TABLE agg_rsi_vs_price (
    date DATE,
    rsi_14 FLOAT,
    [Close] FLOAT
)
WITH (
    LOCATION = 'agg_rsi_vs_price',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO


-- Summary Metrics
CREATE EXTERNAL TABLE agg_summary_metrics (
    date DATE,
    avg_close FLOAT,
    avg_return_1d FLOAT,
    avg_tweet_sentiment FLOAT,
    avg_news_sentiment FLOAT,
    volatility_3d FLOAT,
    avg_rsi_14 FLOAT
)
WITH (
    LOCATION = 'agg_summary_metrics',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO




