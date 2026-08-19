# Stone Paper Scissor Camera Game

This is a two-player Stone Paper Scissor game built with HTML, CSS, JavaScript, and Python. The game uses the webcam to read both players' hand signs and then gives the point to the winner.

Player 1 uses the left side of the camera screen. Player 2 uses the right side.

## Features

- Two-player camera gameplay
- Stone, Paper, and Scissor hand sign detection
- Live score for Player 1 and Player 2
- Round history
- Automatic point update when the result is stable
- Model status shown on the page, so it is easy to check whether the AI model is loaded or fallback mode is running

## Project Files

- `app.py` - Python server and model prediction API
- `index.html` - Main game page
- `style.css` - UI design and responsive layout
- `script.js` - Camera capture, score handling, and game logic
- `keras_model.h5` - Trained hand sign model
- `labels.txt` - Model labels

## Requirements

Install the needed Python packages:

```bash
pip install opencv-python numpy tensorflow tf-keras
```

`tf-keras` is important because the model was exported in an older Keras/Teachable Machine format. Without it, the model may fail to load and the app will show fallback mode.

## How To Run

Open the project folder in terminal and run:

```bash
python app.py
```

If Python 3.13 is used from WindowsApps, run:

```bash
& C:\Users\KiTE\AppData\Local\Microsoft\WindowsApps\python3.13.exe "e:/ips tech intern/task 3/app.py"
```

After running, open:

```text
http://127.0.0.1:8000/index.html
```

The browser may ask for camera permission. Allow it.

## How To Play

1. Click `Start Camera`.
2. Player 1 should stand or show hand on the left side.
3. Player 2 should stand or show hand on the right side.
4. Both players show Stone, Paper, or Scissor.
5. The app detects both signs and decides the winner.
6. The winning player gets a point automatically.
7. Remove or change hands before the next round.

## Debug Notes

On the page, check the model status:

- `MODEL LOADED` means the trained model is running.
- `FALLBACK MODE` means the model did not load, and the app is using basic OpenCV fallback.
- `P1 model:` and `P2 model:` show the model confidence for each player.

If the prediction is wrong, keep the hand clearly inside each player's side. A plain background and good lighting will give better results.

## Common Issues

If the page shows only Stone again and again, the model may not be loaded properly. Check the model status on the page and the terminal error.

If the app shows this kind of error:

```text
DepthwiseConv2D: Unrecognized keyword arguments passed: {'groups': 1}
```

install:

```bash
pip install tf-keras
```

Then stop the server and run `python app.py` again.

## About

This project was made as a simple web-based AI game. The main goal is to make the game easy to play with a friend using just the camera, without pressing move buttons manually.
