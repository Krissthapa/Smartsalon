import numpy as np

def classify_face_shape(landmarks):
    if not landmarks:
        return None
    
    landmarks = np.array(landmarks)
    jaw_width = np.linalg.norm(landmarks[0] - landmarks[16])
    cheek_width = np.linalg.norm(landmarks[1] - landmarks[15])
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])
    
    if face_height / jaw_width > 1.5:
        return 'long'
    elif cheek_width / jaw_width > 1.3:
        return 'square'
    elif jaw_width / cheek_width > 1.5:
        return 'heart'
    else:
        return 'round'
