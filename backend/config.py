# config.py
import os

class Config:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True
