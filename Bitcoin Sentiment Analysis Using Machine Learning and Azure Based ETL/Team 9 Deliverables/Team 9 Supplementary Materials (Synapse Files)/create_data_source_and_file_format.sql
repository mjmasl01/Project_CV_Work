-- Create external data source to connect to your container
CREATE EXTERNAL DATA SOURCE ds562team9datalake_data
WITH (
    LOCATION = 'https://ds562team9datalake.dfs.core.windows.net/gold'
);
GO

-- Create external file format
CREATE EXTERNAL FILE FORMAT ParquetFileFormat
WITH (
    FORMAT_TYPE = PARQUET
);
GO

