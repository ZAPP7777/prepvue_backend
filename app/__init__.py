from flask import Flask
from config import Config
from app.routes import api
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(api)

    logger.info("Flask app initialized successfully.")
    return app
