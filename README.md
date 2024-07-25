# Passive Liveness Detection System

This project is a passive liveness detection system using Vision Transformers (ViT) to differentiate between real and fake users in a video feed. The system uses MTCNN for face detection and a pre-trained Vision Transformer model for liveness detection. The project is built using Flask to provide a web interface and OpenCV for video capture and processing.

## Features

- **Face Detection**: Uses MTCNN for detecting faces in the video stream.
- **Liveness Detection**: Employs a pre-trained Vision Transformer model to classify faces as real or fake.
- **Email Notifications**: Sends email notifications for detected activities.
- **Web Interface**: A Flask-based web interface for interacting with the system.
- **Real-time Processing**: Processes video frames in real-time to evaluate liveness.

## Setup and Installation

### Prerequisites

- Python 3.x
- Flask
- OpenCV
- Torch
- Timm (PyTorch Image Models)
- MTCNN
- Yagmail

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/VickneshB/Liveness_Detection.git
    cd Liveness_Detection
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Pre-trained Model**:
    Ensure you have the pre-trained model `vit_teacher_inc_reduced_lr-7.pth` in the root directory.

4. **Set Up Email Configuration**:
    Replace the placeholder values in `app.py` with your sender email, app password, and receiver email.

### Running the Application
1. **Start the Flask Application**:
    ```bash
    python app.py
    ```

2. **Access the Web Interface**:
    Open a web browser and go to `http://127.0.0.1:5000/`.

## Usage
- **Start the Video Capture**: The video feed will start automatically.
- **Process Frames**: The system will process the video frames in real-time to detect liveness.
- **View Results**: The results of the liveness detection will be displayed on the web interface.
- **Email Notification**: An email notification will be sent if liveness detection is triggered.


## Contributing
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or issues, please reach out to the project maintainer.