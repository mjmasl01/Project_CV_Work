
INSERT INTO agg_crypto_prices_internal (date, avg_open, avg_close, avg_high, avg_low, avg_volume)
SELECT 
    TRY_CAST(date AS DATE),
    TRY_CAST(avg_open AS FLOAT),
    TRY_CAST(avg_close AS FLOAT),
    TRY_CAST(avg_high AS FLOAT),
    TRY_CAST(avg_low AS FLOAT),
    TRY_CAST(avg_volume AS FLOAT)
FROM agg_crypto_prices
WHERE TRY_CAST(date AS DATE) IS NOT NULL;
GO

