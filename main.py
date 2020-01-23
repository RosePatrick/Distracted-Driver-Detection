import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

from utils import get_prediction, upload_cloud, convert_img
from utils import lbl_score, allowed_file

UPLOAD_FOLDER = '/tmp'
bucket_name = 'stable-hybrid-249623.appspot.com'
project_id = 'stable-hybrid-249623'
model_id = 'ICN4772510494057073039'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Upload file to cloud storage
            upload_cloud(bucket_name, filename, filepath)

            # Open image file and convert to required formats
            img, upl_img = convert_img(file)

            # Get prediction from AutoML model
            prediction = get_prediction(img, project_id,  model_id)

            # Convert prediction received from model into readable results
            response = lbl_score(prediction)

            return render_template('predict.html', data=response, img=upl_img)
    return render_template('index.html')
