from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
from process_video import video_inference_with_backgroundremoval, resize_frame, remove_background, load_model
from flask_cors import CORS

app = Flask(__name__) 
CORS(app) #applying CORS setting to the Flask application, making it accesible from any domain

UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'png', 'jpeg', ''}

# Create directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


@app.route('/')
def home():
    return render_template('index.html')

#These variables define the directories where uploaded files are stored and where processed files will be saved, respectively.


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST']) #defines route for uploading files
def upload_file(): #handles upload and processing of files
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Video Proccessing Code
        intermediate_output_path = os.path.join(app.config['PROCESSED_FOLDER'], "intermediate_" + filename)
        processed_filename = "processed_" + filename
        final_output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)  # This remains the same as your original processed_filepath

        try: 
            video_inference_with_backgroundremoval("mobilenet", filepath, intermediate_output_path, final_output_path)
            #Assuming the processing was succesful, construct the URL for the processed video
            processed_video_url = request.url_root + '/processed/'+ processed_filename
            #Return the URL or the filename to the client
            return jsonify({'message': 'File processed succesfully', 'processed_video_url': processed_video_url}), 200
        except Exception as e:
            #handle errors in video processing
            return jsonify({'error': 'Failed to process video', 'details': str(e)}), 500
    else:
        return jsonify({'error': 'File not allowed'}), 400

#routes dynamically serve the requested files from their respective directories

#<filename> is a variable placeholder that represents the name of the file being requested
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#send_from_directory function used to serve files frim both the uploads and processed directories through sepratae routes
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
