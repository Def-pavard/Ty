# Modified main.py
import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import logging
import requests
from dotenv import load_dotenv
from flask import Flask, send_from_directory
from flask_cors import CORS
from src.routes.verifier import verifier_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Activer CORS pour toutes les routes
CORS(app, origins="*")

app.register_blueprint(verifier_bp, url_prefix='/api')

# Removed SQLAlchemy configuration as it's not used in the verifier blueprint
# If needed for other blueprints, re-add it with proper imports

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)