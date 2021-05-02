from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request
from visualize import visualize

# LSP: 1813010
# Model: RetinaNet
# Add. Loss Func.: DR loss
# Data classes: Tie, Door, Laptop

app = Flask(__name__)
run_with_ngrok(app)  # starts ngrok when the app is run


@app.route("/")
def detect():
    images_test_folder = request.args.get('images')
    return visualize(images_test_folder)


app.run()
