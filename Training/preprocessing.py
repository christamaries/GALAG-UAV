import cv2
import numpy as np

FRAME_SIZE = 84
STACK_SIZE = 4


def decode_png(png_bytes):
    """Decode a PNG frame saved by MAME (raw bytes) into a 288x224x3 pixel array."""
    array = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def preprocess_frame(frame):
    """Convert one frame to grayscale and resize it to 84x84."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)


def build_observation(screenshots):
    """Preprocess the last 4 frames and stack them into a (4, 84, 84) uint8 observation."""
    frames = [preprocess_frame(decode_png(png)) for png in screenshots]
    return np.stack(frames).astype(np.uint8)


def normalize(observation):
    """Scale pixels from 0-255 to 0-1. Not used in the environment: Stable-Baselines3 already normalizes images."""
    return observation.astype(np.float32) / 255.0