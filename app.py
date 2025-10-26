import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'

# ---------------- Safe folder creation ----------------
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.isdir(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# ---------------- Point Operations ----------------
def negative_image(img):
    return 255 - img

def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    return np.array(log_image, dtype=np.uint8)

def threshold_image(img, thresh=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

# ---------------- Neighborhood Operations ----------------
def mean_filter(img, k=3):
    return cv2.blur(img, (k, k))

def median_filter(img, k=3):
    return cv2.medianBlur(img, k)

def gaussian_filter(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)

def sobel_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

# ---------------- Flask Routes ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    processed_file = None
    original_file = None
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            original_file = filepath
            img = cv2.imread(filepath)

            # Get operation and parameter
            operation = request.form.get("operation")
            param = request.form.get("param", type=float)

            if operation == "Negative":
                processed = negative_image(img)
            elif operation == "Gamma":
                gamma_val = param if param else 1.0
                processed = gamma_correction(img, gamma=gamma_val)
            elif operation == "Log":
                processed = log_transform(img)
            elif operation == "Threshold":
                thresh_val = int(param) if param else 128
                processed = threshold_image(img, thresh=thresh_val)
            elif operation == "Mean":
                k = int(param) if param else 3
                processed = mean_filter(img, k)
            elif operation == "Median":
                k = int(param) if param else 3
                processed = median_filter(img, k)
            elif operation == "Gaussian":
                k = int(param) if param else 3
                processed = gaussian_filter(img, k)
            elif operation == "Sobel":
                processed = sobel_filter(img)
            else:
                processed = img

            processed_filename = f"{operation}_{file.filename}"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            cv2.imwrite(processed_path, processed)
            processed_file = processed_path

    return render_template("index.html", processed_file=processed_file, original_file=original_file)

if __name__ == "__main__":
    app.run(debug=True)
