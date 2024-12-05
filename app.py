from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from model import load_model, predict_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict_audio(filepath)
        return render_template('result.html', prediction=prediction)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
