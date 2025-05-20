

CREATE EXTERNAL TABLE [dbo].[ext_crypto_tweets] (
    [date] DATE,
    [text] NVARCHAR(MAX),
    [year] INT
)
WITH (
    LOCATION = 'processed_crypto_tweets',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO




CREATE EXTERNAL TABLE [dbo].[ext_crypto_news] (
    [url] NVARCHAR(MAX),
    [title] NVARCHAR(MAX),
    [text] NVARCHAR(MAX),
    [html] NVARCHAR(MAX),
    [year] NVARCHAR(10),
    [author] NVARCHAR(MAX),
    [source] NVARCHAR(MAX),
    [date] DATE
)
WITH (
    LOCATION = 'processed_crypto_news',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO






CREATE EXTERNAL TABLE [dbo].[ext_crypto_prices] (
    [date] DATE,
    [Close] FLOAT,
    [High] FLOAT,
    [Low] FLOAT,
    [Open] FLOAT,
    [Volume] BIGINT
)
WITH (
    LOCATION = 'processed_crypto_prices',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO




CREATE EXTERNAL TABLE [dbo].[ext_sentiment] (
    [_c0] INT,
    [date] DATE,
    [Close] FLOAT,
    [tweet_sentiment] FLOAT,
    [news_sentiment] FLOAT,
    [Close_lag1] FLOAT,
    [return_1d] FLOAT,
    [tweet_sent_roll3] FLOAT,
    [news_sent_roll3] FLOAT,
    [volatility_3d] FLOAT,
    [tweet_sent_roll7] FLOAT,
    [news_sent_roll7] FLOAT,
    [volatility_7d] FLOAT,
    [sma_5] FLOAT,
    [sma_10] FLOAT,
    [rsi_14] FLOAT
)
WITH (
    LOCATION = 'processed_sentiment',
    DATA_SOURCE = ds562team9datalake_data,
    FILE_FORMAT = ParquetFileFormat
);
GO




