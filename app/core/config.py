from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    groq_api_key: str = ""
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./data"
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
