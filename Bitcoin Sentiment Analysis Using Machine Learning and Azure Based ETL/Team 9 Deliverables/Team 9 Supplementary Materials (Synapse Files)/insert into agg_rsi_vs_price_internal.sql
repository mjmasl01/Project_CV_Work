
INSERT INTO agg_rsi_vs_price_internal (date, rsi_14, [Close])
SELECT 
    TRY_CAST(date AS DATE),
    TRY_CAST(rsi_14 AS FLOAT),
    TRY_CAST([Close] AS FLOAT)
FROM agg_rsi_vs_price
WHERE TRY_CAST(date AS DATE) IS NOT NULL;
GO

