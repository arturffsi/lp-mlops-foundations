import os
import pandas as pd

def load_data(source: str, uri: str, sql: str | None = None, redshift_kwargs: dict | None = None) -> pd.DataFrame:
    """
    source: 'parquet' or 'redshift'
    uri: for parquet -> s3://bucket/path/*.parquet
         for redshift -> ignored (use in kwargs if you want)
    sql:  (for redshift) SELECT ...; should return same columns as parquet
    redshift_kwargs: dict with keys like host, port, dbname, user, password, iam/role, etc
    """
    if source == "parquet":
        # supports single file or wildcard prefix
        if uri.endswith(".parquet") and "*" not in uri:
            return pd.read_parquet(uri, engine="pyarrow")
        else:
            # read multiple parquet files into one DF
            import s3fs
            fs = s3fs.S3FileSystem()
            paths = fs.glob(uri)
            dfs = [pd.read_parquet(f"s3://{p}", engine="pyarrow") for p in paths]
            return pd.concat(dfs, ignore_index=True)

    elif source == "redshift":
        import sqlalchemy as sa
        if not sql:
            raise ValueError("Provide SQL for Redshift.")
        # Example for SQLAlchemy + redshift_connector (use your auth method)
        from redshift_connector import connect
        conn = connect(**redshift_kwargs)  # host, database, user, password, port
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()
    else:
        raise ValueError(f"Unknown source: {source}")