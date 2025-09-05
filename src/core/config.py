from typing import Any, Dict, Optional

import boto3
from pydantic.v1 import BaseSettings, PostgresDsn, validator


class SyncPostgresDsn(PostgresDsn):
    allowed_schemes = {"postgresql", "postgresql+psycopg2", "postgresql+pg8000"}

class Settings(BaseSettings):
    # Root dir
    ROOT_DIR: str = "/app"
    # Default datadir 
    DATA_DIR: str = "/app/src/data"
    # Input datadir 
    INPUT_DATA_DIR: str = "/app/src/data/input"
    # Output datadir
    OUTPUT_DATA_DIR: str = "/app/src/data/output"
    # Data config dir
    CONFIG_DIR: str = "/app/src/config"


    # Local Database Settings
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_PORT: Optional[str] = None
    LOCAL_DATABASE_URI: Optional[SyncPostgresDsn] = None
    @validator("LOCAL_DATABASE_URI", pre=True)
    def assemble_local_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return SyncPostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
            port=values.get("POSTGRES_PORT")
        )

    # Raw Database Settings
    POSTGRES_USER_RD: Optional[str] = None
    POSTGRES_PASSWORD_RD: Optional[str] = None
    POSTGRES_HOST_RD: Optional[str] = None
    POSTGRES_DB_RD: Optional[str] = None
    POSTGRES_PORT_RD: Optional[str] = None
    RAW_DATABASE_URI: Optional[SyncPostgresDsn] = None
    @validator("RAW_DATABASE_URI", pre=True)
    def assemble_remote_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return SyncPostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER_RD"),
            password=values.get("POSTGRES_PASSWORD_RD"),
            host=values.get("POSTGRES_HOST_RD"),
            path=f"/{values.get('POSTGRES_DB_RD') or ''}",
            port=values.get("POSTGRES_PORT_RD")
        )
    POSTGRES_SCHEMA_RD: Optional[str] = "basic"

    # GOAT Database Settings
    POSTGRES_USER_GOAT: Optional[str] = None
    POSTGRES_PASSWORD_GOAT: Optional[str] = None
    POSTGRES_HOST_GOAT: Optional[str] = None
    POSTGRES_DB_GOAT: Optional[str] = None
    POSTGRES_PORT_GOAT: Optional[str] = None
    GOAT_DATABASE_URI: Optional[SyncPostgresDsn] = None
    #@validator("GOAT_DATABASE_URI", pre=True)
    def assemble_goat_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return SyncPostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER_GOAT"),
            password=values.get("POSTGRES_PASSWORD_GOAT"),
            host=values.get("POSTGRES_HOST_GOAT"),
            path=f"/{values.get('POSTGRES_DB_GOAT') or ''}",
            port=values.get("POSTGRES_PORT_GOAT")
        )
    POSTGRES_SCHEMA_GOAT: Optional[str] = "basic"

    # CityGML Settings
    POSTGRES_USER_3DCITY: Optional[str] = None
    POSTGRES_PASSWORD_3DCITY: Optional[str] = None
    POSTGRES_HOST_3DCITY: Optional[str] = None
    POSTGRES_DB_3DCITY: Optional[str] = None
    POSTGRES_PORT_3DCITY: Optional[str] = None
    CITYGML_DATABASE_URI: Optional[SyncPostgresDsn] = None
    #@validator("CITYGML_DATABASE_URI", pre=True)
    def assemble_citygml_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return SyncPostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER_3DCITY"),
            password=values.get("POSTGRES_PASSWORD_3DCITY"),
            host=values.get("POSTGRES_HOST_3DCITY"),
            path=f"/{values.get('POSTGRES_DB_3DCITY') or ''}",
            port=values.get("POSTGRES_PORT_3DCITY")
        )
    AWS_BUCKET_CITYGML: Optional[str] = None
    
    # AWS Client 
    AWS_BUCKET_NAME: str = None
    AWS_ACCESS_KEY_ID: str = None
    AWS_SECRET_ACCESS_KEY: str = None
    AWS_DEFAULT_REGION: str = None
    S3_CLIENT: Optional[Any] = None
    #@validator("S3_CLIENT", pre=True)
    def assemble_s3_client(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        # Key, secret and token must be set manually if using short-term credentials
        return boto3.client(
            's3',
            aws_access_key_id=values.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=values.get("AWS_ACCESS_KEY_ID"),
            # aws_session_token="",
            region_name=values.get("AWS_DEFAULT_REGION")
        )
    GITHUB_ACCESS_TOKEN: str = None
    
    # R5
    R5_FRONTEND_HOST: str = None
    R5_FRONTEND_PORT: str = None
    R5_BACKEND_HOST: str = None
    R5_BACKEND_PORT: str = None
    R5_AUTHORIZATION: str = None
    @validator("R5_AUTHORIZATION", pre=True)
    def r5_authorization(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if v:
            return f"Basic {v}="
        return None
    
settings = Settings()
