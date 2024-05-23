# HoloConnect 👋: Hologram Video Generator
As an international student, I often felt disconnected from my family and friends due to the physical distance. I wanted to reimagine the user experience and interaction of online communication to make it feel more real and personal. This inspired me to create HoloConnect, a web app that processes videos for holographic display to provide a more immersive and lifelike way of connecting with loved ones.

[![Demo](https://img.youtube.com/vi/8fHARcOWop8/0.jpg)](https://www.youtube.com/watch?v=8fHARcOWop8)

### What it does
HoloConnect allows users to upload any video or image from their files or camera roll. The app processes the media by removing the background and replicating the foreground in a circular axis, creating a Pepper's Ghost illusion when viewed through a hologram display prism. This makes the holographic experience feel realistic and engaging.

### How I Built It
#### Frontend: 
Developed with HTML, CSS, and JavaScript to handle user interactions and video uploads.

- index.html: Provides the structure and interface for video uploads and display​​.
- style.css: Styles the user interface to enhance usability and aesthetics.
- script.js: Manages the upload process, shows a loading spinner, and handles the display of the processed video​​.

#### Backend: 
Implemented using Flask (Python) to handle video processing.

- process_video.py:
  - Utilizes semantic segmentation to identify pixels from the background class from each frame of the video, creating a binary mask to isolate the foreground. This mask is applied to each frame of the video, isolating the subject from the background.The ML model used for semantic segmentation in this project is Mobile DeepLabV3 from TensorFlow.
  - The processed frames are then rotated and arranged in a circular pattern. This arrangement is crucial for creating the Pepper's Ghost illusion when viewed through a hologram prism.
  - The audio from the original video is retained and synchronized with the processed video to ensure a cohesive holographic experience.
