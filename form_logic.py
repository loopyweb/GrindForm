"""Pushup form analysis and rep counting.

Pure Python with no camera, OpenCV or MediaPipe imports, so it can be unit tested.
"""
import math

# Elbow angle (degrees) at the top and bottom of a rep.
UP_ANGLE = 160
DOWN_ANGLE = 90
# Coming back up after bending below this, without reaching DOWN_ANGLE, is a partial rep.
PARTIAL_ANGLE = 140
# Shoulder-hip-knee angle below this means the hips are sagging or piked.
HIP_MIN_ANGLE = 160
# Landmarks MediaPipe is less sure about than this are treated as not visible.
MIN_VISIBILITY = 0.5

MSG_READY = "Get ready!"
MSG_GOOD_FORM = "Form looks good!"
MSG_GOOD_REP = "Good rep!"
MSG_BACK = "Keep your back straight!"
MSG_NOT_COUNTED = "Not counted: keep your back straight"
MSG_DEEPER = "Go deeper!"
MSG_NOT_VISIBLE = "Turn sideways so your whole body is visible"

# MediaPipe Pose landmark indices: shoulder, elbow, wrist, hip, knee.
JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee")
LEFT_SIDE = (11, 13, 15, 23, 25)
RIGHT_SIDE = (12, 14, 16, 24, 26)


def calculate_angle(a, b, c):
    """Angle ABC in degrees (0-180), with B as the vertex."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    return 360 - angle if angle > 180 else angle


def side_points(landmarks, width, height):
    """Pixel coordinates of the joints on the side of the body facing the camera.

    `landmarks` is a sequence of objects with normalized `x`, `y` and `visibility`
    (MediaPipe's pose landmarks). Coordinates are scaled to pixels because MediaPipe
    normalizes x by width and y by height, which distorts angles on non-square frames.
    Returns a dict keyed by JOINTS, or None if neither side is clearly visible.
    """
    def visibility(indices):
        return min(landmarks[i].visibility for i in indices)

    best = max((LEFT_SIDE, RIGHT_SIDE), key=visibility)
    if visibility(best) < MIN_VISIBILITY:
        return None
    return {name: (landmarks[i].x * width, landmarks[i].y * height) for name, i in zip(JOINTS, best)}


class PushupCounter:
    """Counts a rep when the elbow goes from straight to below DOWN_ANGLE and back up.

    A rep doesn't count if the hips broke alignment at any point while the arms were bent.
    """

    def __init__(self):
        self.reps = 0
        self.stage = None  # "up" or "down"
        self.feedback = MSG_READY
        self._lowest = 180.0
        self._form_broken = False

    def update(self, elbow_angle, hip_angle):
        hips_ok = hip_angle >= HIP_MIN_ANGLE
        if not hips_ok:
            self.feedback = MSG_BACK
            if elbow_angle <= UP_ANGLE:
                self._form_broken = True
        elif self.feedback == MSG_BACK:
            self.feedback = MSG_GOOD_FORM

        if self.stage == "up" and elbow_angle < DOWN_ANGLE:
            self.stage = "down"
        self._lowest = min(self._lowest, elbow_angle)

        if elbow_angle > UP_ANGLE:
            if self.stage == "down":
                if self._form_broken:
                    self.feedback = MSG_NOT_COUNTED
                else:
                    self.reps += 1
                    self.feedback = MSG_GOOD_REP
            elif self.stage == "up" and self._lowest < PARTIAL_ANGLE:
                self.feedback = MSG_DEEPER
            elif self.stage is None and hips_ok:
                self.feedback = MSG_GOOD_FORM
            self.stage = "up"
            self._lowest = elbow_angle
            self._form_broken = False
        return self.feedback

    def update_from_points(self, points):
        elbow = calculate_angle(points["shoulder"], points["elbow"], points["wrist"])
        hip = calculate_angle(points["shoulder"], points["hip"], points["knee"])
        return self.update(elbow, hip)
