from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64
import json
import os

app = Flask(__name__)

ENCODING_DB = 'face_db.json'

# --- Utility Function to Decode Base64 Image ---
def read_image_from_base64(data):
    try:
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Failed to decode image:", e)
        return None

# --- Save Face Encoding to JSON File ---
def save_encoding(name, encoding):
    if os.path.exists(ENCODING_DB):
        with open(ENCODING_DB, 'r') as f:
            db = json.load(f)
    else:
        db = []

    db.append({
        'name': name,
        'encoding': encoding
    })

    with open(ENCODING_DB, 'w') as f:
        json.dump(db, f)

# --- Register New Face ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    image_b64 = data.get('image')
    name = data.get('name')

    if not image_b64 or not name:
        return jsonify({'status': 'error', 'message': 'Missing name or image'}), 400

    img = read_image_from_base64(image_b64)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        return jsonify({'status': 'error', 'message': 'No face found'}), 400

    encoding_list = encodings[0].tolist()
    save_encoding(name, encoding_list)

    return jsonify({
        'status': 'success',
        'message': f'Face registered for {name}',
        'encoding': encoding_list  # Include this in the response
    }), 200


# --- Recognize Faces ---
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_b64 = data.get('image')

    if not image_b64:
        return jsonify({'status': 'error', 'message': 'Missing image'}), 400

    img = read_image_from_base64(image_b64)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

    face_encodings = face_recognition.face_encodings(img)

    if not os.path.exists(ENCODING_DB):
        return jsonify({'status': 'error', 'message': 'Encoding database not found'}), 500

    with open(ENCODING_DB, 'r') as f:
        db = json.load(f)

    known_encodings = [np.array(entry['encoding']) for entry in db]
    known_names = [entry['name'] for entry in db]

    matches_info = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) == 0:
            # No known faces in DB
            matches_info.append({
                'name': 'Unknown',
                'confidence': None,
                'encoding': face_encoding.tolist()

            })
            continue

        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        # Lower distance means better match — you can define a threshold, e.g., 0.6
        threshold = 0.6
        if best_distance <= threshold:
            name = known_names[best_match_index]
        else:
            name = 'Unknown'

        matches_info.append({
            'name': name,
            'confidence': float(best_distance),
            'encoding': face_encoding.tolist()

        })

    return jsonify({
        'status': 'success',
        'matches': matches_info
    }), 200


# --- Start Flask Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

