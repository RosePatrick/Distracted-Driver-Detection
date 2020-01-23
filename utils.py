import base64
from PIL import Image
from io import BytesIO
from google.cloud import storage
from google.cloud import automl_v1beta1

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# define class labels
class_dict = {'c0': 'safe driving', 'c1': 'texting - right',
              'c2': 'talking on the phone - right',
              'c3': 'texting - left', 'c4': 'talking on the phone - left',
              'c5': 'operating the radio', 'c6': 'drinking',
              'c7': 'reaching behind', 'c8': 'hair and makeup',
              'c9': 'talking to passenger'}


# check if file type is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# get image prediction from API
def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned


# convert predicted label (e.g. c0, c1) to text (e.g. 'safe driving')
def get_label(pred):
    label = class_dict[pred]
    return label


# Upload file to cloud storage
def upload_cloud(bucket_name, filename, filepath):
    name = '/{}/{}'.format(bucket_name, filename)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = storage.Blob(name, bucket)
    blob.upload_from_filename(filepath)
    return


# open image file and convert to required formats
def convert_img(file):
    image = Image.open(file)
    img = BytesIO()
    image.save(img, format='JPEG')
    img.seek(0)
    # image data for class prediction
    img = img.getvalue()
    # image data for display on HTML page
    upl_img = base64.b64encode(img)
    upl_img = upl_img.decode('ascii')
    return img, upl_img


# extract label and score from AutoML API
# convert to readable label and return response as a dict
def lbl_score(prediction):
    pred_label = prediction.payload[0].display_name
    lbl = get_label(pred_label)
    score = prediction.payload[0].classification.score
    response = {
        'prediction': {
            'label': lbl,
            'score': score
        }
    }
    return response
