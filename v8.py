import cv2
import imutils
import multiprocessing as mp
import ffmpeg
from datetime import datetime, timedelta


# Function to extract the creation date using ffmpeg (optional)
def get_video_creation_date(video_path):
    """
    Extracts the video creation date from metadata using ffmpeg.

    Args:
        video_path (str): Path to the video file.

    Returns:
        datetime: The creation date of the video if found, otherwise None.
    """
    try:
        probe = ffmpeg.probe(video_path)
        creation_time = None
        for stream in probe['format']['tags']:
            if 'creation_time' in stream:
                creation_time = probe['format']['tags']['creation_time']
                break

        if creation_time:
            return datetime.strptime(creation_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            return None
    except ffmpeg.Error as e:
        print(f"Error extracting metadata: {e}")
        return None


# Streamer Process
def streamer(video_path, detector_queue):
    """
    Streams frames from the video and sends them to the detector process.

    Args:
        video_path (str): Path to the video file.
        detector_queue (Queue): Queue to send frames to the detector process.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Extract the video's frame rate

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        detector_queue.put(frame)  # Send the frame to the detector
    cap.release()

    # Send signal to stop the detector when video ends
    detector_queue.put(None)


# Detector Process
def detector(detector_queue, presenter_queue):
    """
    Detects motion by comparing consecutive frames and sends results to the presenter process.

    Args:
        detector_queue (Queue): Queue to receive frames from the streamer process.
        presenter_queue (Queue): Queue to send frames and detection results to the presenter process.
    """
    prev_frame = None

    while True:
        frame = detector_queue.get()  # Receive the frame from the streamer
        if frame is None:
            # Send signal to stop the presenter
            presenter_queue.put((None, None))
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = gray_frame
            continue

        # Detect motion by finding the difference between the current and previous frame
        diff = cv2.absdiff(gray_frame, prev_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of the motion
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Send frame and detections to the presenter
        presenter_queue.put((frame, cnts))

        prev_frame = gray_frame


# Presenter Process with Blurring and Running Timestamp
def presenter(presenter_queue, creation_date, fps):
    """
    Receives frames from the detector, applies blurring to detected motion areas, and displays the frames.

    Args:
        presenter_queue (Queue): Queue to receive frames and detections from the detector process.
        creation_date (datetime): The video's creation date, to be used as the base time.
        fps (int): Frame rate of the video to ensure smooth playback.
    """
    cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)

    MIN_BOX_SIZE = (50, 50)  # Minimum box width and height
    delay_between_frames = int(1000 / fps)  # Calculate delay between frames based on FPS

    # Track the time passed based on the frame rate and video creation time
    frame_duration = timedelta(seconds=1 / fps)  # Time added per frame
    current_time = creation_date  # Start with the creation date

    while True:
        frame, cnts = presenter_queue.get()  # Receive the frame and detections from the detector
        if frame is None:
            break  # Exit when signal to stop is received

        # Apply blurring and resizing to detected regions
        for contour in cnts:
            if cv2.contourArea(contour) < 650:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)

            # Enforce minimum size for the box
            if w < MIN_BOX_SIZE[0]:
                w = MIN_BOX_SIZE[0]
            if h < MIN_BOX_SIZE[1]:
                h = MIN_BOX_SIZE[1]

            # Blur the detected region
            roi = frame[y:y + h, x:x + w]
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
            frame[y:y + h, x:x + w] = blurred_roi  # Replace the original region with the blurred region

            # Draw a rectangle around the blurred region with a fixed minimum size
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the running timestamp (incrementing from the creation date)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Update the time for the next frame
        current_time += frame_duration

        # Display the frame with motion detection and blurring
        cv2.imshow("Motion Detection", frame)
        # Makes the video smother.
        if cv2.waitKey(delay_between_frames) & 0xFF == ord('q'):
            break

    # Close the window when the video ends
    cv2.destroyAllWindows()


# Main Function with metadata option
def main(video_path, use_metadata=True):
    """
    Main function that initializes the streamer, detector, and presenter processes.

    Args:
        video_path (str): Path to the video file.
        use_metadata (bool): Whether to attempt extracting metadata for the creation date.
    """
    # If use_metadata is True, try to get the video creation date
    if use_metadata:
        creation_date = get_video_creation_date(video_path)
        if creation_date is None:
            print("Metadata not found. Using today's date instead.")
            creation_date = datetime.now()
    else:
        # If metadata is disabled, just use today's date
        creation_date = datetime.now()

    # Open the video to extract frame rate
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Create queues for communication between processes
    detector_queue = mp.Queue(maxsize=5)
    presenter_queue = mp.Queue(maxsize=5)

    # Create separate processes for each component
    streamer_process = mp.Process(target=streamer, args=(video_path, detector_queue))
    detector_process = mp.Process(target=detector, args=(detector_queue, presenter_queue))
    presenter_process = mp.Process(target=presenter, args=(presenter_queue, creation_date, fps))

    # Start all processes
    streamer_process.start()
    detector_process.start()
    presenter_process.start()

    # Wait for processes to finish
    streamer_process.join()
    detector_process.join()
    presenter_process.join()

    # Print final message when all processes are done
    print("All processes are done. System has shut down.")


if __name__ == "__main__":
    video_path = "vid2.mp4"
    main(video_path, use_metadata=True)  # Set to False to avoid reading metadata

# Google Python Style Guide
# https://stackoverflow.com/questions/3898572/what-are-the-most-common-python-docstring-formats

# References:
# https://www.youtube.com/watch?v=lBMosWq8NKo
# https://www.youtube.com/watch?v=T-7OSD5a-88
# https://github.com/Gupu25/PeopleCounter
# https://www.youtube.com/watch?v=aEcBnD80nLg
# https://www.youtube.com/watch?v=caKnQlCMIYI
