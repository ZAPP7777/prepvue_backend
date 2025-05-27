import os

class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", 5001))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")