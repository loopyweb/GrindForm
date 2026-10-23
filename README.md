# GrindForm: AI Pushup Form Checker & Tracker

GrindForm uses computer vision to analyze your pushup form in real time. It tracks your joints
with Google's MediaPipe Pose Landmarker and OpenCV, counts your reps, and gives instant feedback
on your posture.

## Features

- **Rep counter:** a rep counts when your arms go from straight (elbow above 160°) to below 90°
  and back up.
- **Form analysis:** checks elbow depth (the 90° rule) and hip alignment (shoulder–hip–knee
  angle of at least 160°).
- **Real-time feedback:** on-screen prompts such as "Go deeper!", "Keep your back straight!"
  and "Good rep!". A rep where your hips sagged or piked while your arms were bent is not counted.
- **Either side:** uses whichever side of your body faces the camera more clearly.

## Installation

Requires Python 3 (tested on 3.13) and a webcam.

```bash
git clone https://github.com/loopyweb/GrindForm.git
cd GrindForm
pip install -r requirements.txt
```

The first run downloads MediaPipe's pose model (about 6 MB) into `models/`.

## Usage

```bash
python main.py
```

Set up so the camera sees your whole body from the side, then start your pushups.
Press **q** to quit; the total is printed when you exit.

Options:

- `--camera N`: use webcam number N (default 0)
- `--video PATH`: analyze a recorded video instead of the webcam
- `--model PATH`: use a different pose landmarker `.task` model

## Project layout

- `main.py`: camera loop, pose detection and on-screen display
- `form_logic.py`: angle math and the rep-counting state machine (no camera or MediaPipe needed)
- `test_form_logic.py`: unit tests

## Tests

```bash
python -m unittest -v
```
