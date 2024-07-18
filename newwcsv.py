import torch
import numpy as np
import cv2
import easyocr
import re
from time import time
import csv
from torchvision import transforms
from PIL import Image
from datetime import datetime


class CarDetection:
    def __init__(self, capture_index, model_name, csv_filename='output.csv'):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cpu'
        self.ocr_reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
        self.csv_filename = csv_filename
        print("Using device:", self.device)

    def setup_csv(self):
        # Create the CSV file with headers if it doesn't exist
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Detected Text', 'Confidence', 'Timestamp'])

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def sharpen_image(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return sharpened

    def apply_clahe(self, gray_img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_img)

    def preprocess_for_ocr(self, cropped_img):
        sharpened = self.sharpen_image(cropped_img)
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe_img = self.apply_clahe(blurred)
        alpha = 2.0  # Contrast control
        beta = -50  # Brightness control
        enhanced_img = cv2.convertScaleAbs(clahe_img, alpha=alpha, beta=beta)
        binary = cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        processed_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return processed_img

    def validate_indian_license_plate(self, text):
        patterns = [
            re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]\s?\d{4}$'),
            re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{1,2}\s?\d{4}$')
        ]
        return any(pattern.match(text) for pattern in patterns)

    def correct_common_errors(self, text):
        text = text.strip()
        text = re.sub(r'^[^\w]+', '', text)
        text = re.sub(r'\s+', '', text)
        corrected_text = []
        length = len(text)
        patterns = {
            10: ["L", "L", "D", "D", "L", "L", "D", "D", "D", "D"],
            9: ["L", "L", "D", "D", "L", "D", "D", "D", "D"]
        }
        expected_pattern = patterns.get(length, [])

        for i, char in enumerate(text):
            if i < len(expected_pattern):
                expected_type = expected_pattern[i]
                if expected_type == "L":
                    if char.lower() in '0123456789':
                        corrected_char = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G'}.get(char, char)
                    elif char == 'F' and i == 0:
                        corrected_char = 'P'
                    else:
                        corrected_char = char.upper()
                elif expected_type == "D":
                    if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        corrected_char = {'O': '0', 'I': '1', 'T': '1', 'D': '0', 'B': '8', 'Z': '2'}.get(char, char)
                    else:
                        corrected_char = char
            else:
                corrected_char = char
            corrected_text.append(corrected_char)

        return ''.join(corrected_text)

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cropped_img = frame[y1:y2, x1:x2]
                cropped_img = cv2.resize(cropped_img, (430, 210))
                cv2.imshow('cropped', cropped_img)
                preprocessed_img = self.preprocess_for_ocr(cropped_img)
                cv2.imshow('Preprocessed for OCR', preprocessed_img)
                ocr_results = self.ocr_reader.readtext(preprocessed_img)
                for (bbox, text, prob) in ocr_results:
                    if len(text) > 4 and prob >= 0.5:
                        corrected_text = self.correct_common_errors(text)
                        if self.validate_indian_license_plate(corrected_text):
                            text_x, text_y = x1, y1 - 10
                            cv2.putText(frame, corrected_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            print(f"Corrected and Valid Text: {corrected_text} with confidence: {prob}")
                            self.write_to_csv(corrected_text, prob)
                        else:
                            print(f"Invalid detected text after correction: {corrected_text}")
        return frame

    def write_to_csv(self, text, confidence):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([text, confidence, timestamp])

    def class_to_label(self, class_idx):
        return self.classes[int(class_idx)]

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        print(f"Attempting to open video source: {self.capture_index}")
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {self.capture_index}")
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (416, 416))
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            cv2.imshow('yolov5 detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
detector = CarDetection(capture_index='car.mp4', model_name='bestl.pt', csv_filename='output.csv')
detector.setup_csv()
detector()
