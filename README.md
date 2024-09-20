# Video Motion Detection with Metadata Timestamp

This Python project detects motion in video frames and dynamically applies a running timestamp based on the video creation date extracted from metadata. If the metadata is not available, it defaults to using the current date. Detected motion areas are blurred to highlight privacy or to focus on specific regions.

## Features

- **Motion Detection**: Detects motion in video frames by comparing consecutive frames.
- **Blurring Motion Regions**: Automatically blurs areas where motion is detected.
- **Dynamic Timestamp**: Displays a running timestamp starting from the video's creation date (if available in metadata) or defaults to the current date.
- **Smooth Frame Rate Handling**: Ensures smooth playback by adjusting frame display timing based on the video's frame rate.
- **Parallel Processing**: Utilizes Python's `multiprocessing` to run the streamer, detector, and presenter processes in parallel for efficient operation.

## Requirements

- **Python 3.x**
- **FFmpeg**
- **Libraries**: 
  - OpenCV (`opencv-python`)
  - FFmpeg Python wrapper (`ffmpeg-python`)
  - Imutils (`imutils`)

### Install Dependencies

Install the required Python libraries using `pip`:

```bash
pip install opencv-python ffmpeg-python imutils
