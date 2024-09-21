import cv2
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta

# Parameters for Motion History Image
MHI_DURATION = 0.5  # Duration to keep motion in the history
MAX_TIME_DELTA = 0.2  # Maximum allowed time delta between frames
MIN_TIME_DELTA = 0.05  # Minimum allowed time delta for the gradient

# Initialize the Motion History Image (MHI)
def init_motion_history(frame_shape):
    return np.zeros(frame_shape, dtype=np.float32)

# Update the Motion History Image manually
def update_motion_history(motion_mask, mhi, timestamp):
    # Motion regions are set to the current timestamp
    mhi[motion_mask > 0] = timestamp
    # Decay the history over time
    mhi[mhi < (timestamp - MHI_DURATION)] = 0

# Visualize the Motion History Image
def visualize_motion(mhi, timestamp):
    # Normalize MHI for visualization (motion is lighter when recent)
    vis = np.uint8(np.clip((mhi - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
    return vis

# Streamer Process
def streamer(video_path, detector_queue):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        detector_queue.put(frame)  # Send the frame to the detector
    cap.release()

    detector_queue.put(None)

# Detector Process with MHI and motion blurring
def detector(detector_queue, presenter_queue):
    prev_frame = None
    motion_history = None  # Initialize later based on frame size
    timestamp = 0  # Keep track of time

    while True:
        frame = detector_queue.get()  # Receive the frame from the streamer
        if frame is None:
            presenter_queue.put((None, None))
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)  # Smoothing to reduce noise

        if motion_history is None:
            motion_history = init_motion_history(gray_frame.shape)

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        # Compute difference between current and previous frames
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Update motion history
        timestamp += 1 / 30  # Assuming 30 FPS
        update_motion_history(motion_mask, motion_history, timestamp)

        # Visualize motion history image
        motion_vis = visualize_motion(motion_history, timestamp)

        # Find contours of motion regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Blur motion regions
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Ignore small motions
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            # Blur the detected region in the frame
            roi = frame[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
            frame[y:y+h, x:x+w] = blurred_roi  # Replace with blurred region
            # Draw a rectangle around the blurred region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        presenter_queue.put((frame, motion_vis))

        prev_frame = gray_frame

# Presenter Process
def presenter(presenter_queue):
    cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)

    while True:
        frame, motion_vis = presenter_queue.get()
        if frame is None:
            break

        # Show both the original frame with blurred moving objects and the motion history image
        combined = np.hstack((frame, cv2.cvtColor(motion_vis, cv2.COLOR_GRAY2BGR)))

        # Display the frame with motion detection and blurring
        cv2.imshow("Motion Detection", combined)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Main Function
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    detector_queue = mp.Queue(maxsize=5)
    presenter_queue = mp.Queue(maxsize=5)

    streamer_process = mp.Process(target=streamer, args=(video_path, detector_queue))
    detector_process = mp.Process(target=detector, args=(detector_queue, presenter_queue))
    presenter_process = mp.Process(target=presenter, args=(presenter_queue,))

    streamer_process.start()
    detector_process.start()
    presenter_process.start()

    streamer_process.join()
    detector_process.join()
    presenter_process.join()

if __name__ == "__main__":
    video_path = "vid3.mp4"
    main(video_path)
