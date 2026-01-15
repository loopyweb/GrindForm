import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Logic Variables
counter = 0 
stage = None # "up" or "down"
feedback = "Get Ready!"

def calculate_angle(a, b, c):
    a = np.array(a) # First (Shoulder)
    b = np.array(b) # Mid (Elbow)
    c = np.array(c) # End (Wrist)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle 

cap = cv2.VideoCapture(0)

# Setup MediaPipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for Elbow Angle
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Get coordinates for Hip/Back Alignment
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            hip_angle = calculate_angle(shoulder, hip, knee)

            # Pushup Counter & Form Logic
            # 1. Check for Hip Alignment (Plank form)
            if hip_angle < 160:
                feedback = "Keep your back straight!"
            else:
                feedback = "Form looks good!"

                # 2. Pushup State Machine
                if elbow_angle > 160:
                    stage = "up"
                if elbow_angle < 90 and stage =='up':
                    stage = "down"
                    counter += 1
                    print(f"Rep Count: {counter}")

        except:
            pass

        # UI Overlay: Render Rep Counter and Feedback
        cv2.rectangle(image, (0,0), (250, 80), (245, 117, 16), -1)
        cv2.putText(image, f'REPS: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'STAGE: {stage}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, feedback, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw body landmarks on screen
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)               
        
        cv2.imshow('Pushup Form Checker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()