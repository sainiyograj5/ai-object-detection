# AI Object Detection

A web application for real-time object detection using YOLOv11, built with FastAPI backend and vanilla JavaScript frontend.

## Features

- 🔐 User authentication (JWT-based)
- 🤖 YOLOv11 object detection
- 📊 Real-time detection results with confidence scores
- 🎨 Beautiful, responsive UI
- 📸 Image upload and processing

## Technologies Used

- **Backend**: FastAPI, Python
- **ML Model**: YOLOv11 (Ultralytics)
- **Authentication**: JWT tokens
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, PIL

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/YOUR_USERNAME/ai-object-detection.git
   cd ai-object-detection
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Download YOLOv11 model:
```bash
   # The model will be downloaded automatically on first run
```

4. Run the backend:
```bash
   python main.py
```

5. Open `index.html` in your browser or serve it using a local server.

## Usage

1. **Sign up**: Create a new account
2. **Login**: Use your credentials to log in
3. **Upload**: Select an image to detect objects
4. **Results**: View detected objects with bounding boxes and confidence scores

## API Endpoints

- `POST /signup` - Create new user account
- `POST /login` - Login and get JWT token
- `POST /uploadimage` - Upload image for detection (requires auth)
- `GET /health` - Health check endpoint

## Default Test User

- Username: `testuser`
- Password: `testpass123`

## License

MIT License