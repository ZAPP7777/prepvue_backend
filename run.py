from app import create_app
from config import Config

app = create_app()

if __name__ == '__main__':
    app.run(port=Config.PORT, host="0.0.0.0")