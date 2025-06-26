from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # Application settings
    app_name: str = Field(default="Hybrid Search RAG API")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")
    
    # Azure OpenAI settings
    azure_openai_endpoint: str = Field(...)
    azure_openai_api_key: str = Field(...)
    azure_openai_api_version: str = Field(default="2024-02-01")
    azure_embedding_deployment: str = Field(...)
    azure_llm_deployment: str = Field(...)
    
    # MinIO settings
    minio_endpoint: str = Field(default="minio:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket_name: str = Field(default="documents")
    minio_secure: bool = Field(default=False)
    
    # Qdrant settings
    qdrant_host: str = Field(default="qdrant")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="hybrid_search")
    qdrant_grpc_port: int = Field(default=6334)
    
    # Chunking settings
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    
    # Search settings
    top_k_results: int = Field(default=10)
    hybrid_alpha: float = Field(default=0.7)  # Weight for dense search
    rerank_top_k: int = Field(default=20)
    
    # Embedding settings
    embedding_batch_size: int = Field(default=30)  # Reduced to avoid rate limits
    embedding_dimension: int = Field(default=3072)  # text-embedding-3-large dimension
    
    # Retry settings
    max_retries: int = Field(default=3)
    initial_retry_delay: float = Field(default=1.0)
    max_retry_delay: float = Field(default=60.0)
    retry_multiplier: float = Field(default=2.0)
    
    # Rate limiting
    embeddings_per_minute: int = Field(default=300)  # Conservative limit for Azure OpenAI
    
    # LLM settings
    llm_system_prompt: str = Field(
        default="""You are a professional assistant with access to internal company data and various connected tools. Your task is to determine the appropriate data based on the user's request.

## Important rules you must always follow:
* Verify data: Always compare the determined data with the original user request. Never pass on information that you have not clearly obtained from a verified data source. Do not invent or guess anything.
* Numeric values: For numbers (e.g. prices), use exclusively the values that come directly from the tools used.
* Justify answers: Always provide a comprehensible justification for your answer, but only based on the available information. Do not invent explanations.
* Cite sources: If you have a source in the source field, always include it in your answer. Use the format: Source: <source>."""
    )
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


settings = Settings()