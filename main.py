from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
from io import BytesIO
import os
import json

# JWT Configuration
SECRET_KEY = "dio2hynCfah80TaVokbOVKMYy7u9eOxzoGdtA8Fexnta73MHqavuWbnAXxEi9jxs"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI(title="AI Object Detection API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Load YOLOv11 model
model = YOLO('yolo11n.pt')

# File-based user storage
USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    # Create default user if file doesn't exist
    default_users = {
        "testuser": {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": pwd_context.hash("testpass123"),
            "created_at": datetime.now().isoformat()
        }
    }
    save_users(default_users)
    return default_users

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# Load users at startup
fake_users_db = load_users()

# Pydantic Models
class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    detections: List[DetectionResult]
    image_with_boxes: str
    total_objects: int
    processing_time: float

# Authentication Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# API Endpoints
@app.post("/signup")
async def signup(user: UserSignup):
    """Signup endpoint to create new user account"""
    global fake_users_db
    
    # Reload users to get latest data
    fake_users_db = load_users()
    
    # Check if username already exists
    if user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Check if email already exists
    for existing_user in fake_users_db.values():
        if existing_user.get("email") == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Validate password length
    if len(user.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )
    
    # Validate username length
    if len(user.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    
    # Create new user
    fake_users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hash_password(user.password),
        "created_at": datetime.now().isoformat()
    }
    
    # Save to file
    save_users(fake_users_db)
    
    return {
        "message": "Account created successfully",
        "username": user.username,
        "email": user.email
    }

@app.post("/login", response_model=Token)
async def login(user: UserLogin):
    """Login endpoint to get JWT token"""
    global fake_users_db
    
    # Reload users to get latest data
    fake_users_db = load_users()
    
    db_user = fake_users_db.get(user.username)
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/uploadimage", response_model=DetectionResponse)
async def upload_image(
    file: UploadFile = File(...),
    username: str = Depends(verify_token)
):
    """
    Upload image endpoint with YOLOv11 object detection
    Requires JWT authentication
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        # Start timing
        start_time = datetime.now()
        
        # Run YOLOv11 inference
        results = model(img)
        
        # Process results
        detections = []
        annotated_img = img.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Add to detections list
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Draw bounding box on image
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_img, label, (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "detections": detections,
            "image_with_boxes": f"data:image/jpeg;base64,{img_base64}",
            "total_objects": len(detections),
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "signup": "/signup",
            "login": "/login",
            "upload": "/uploadimage"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/users/count")
async def get_user_count():
    """Get total number of registered users (for testing)"""
    users = load_users()
    return {"total_users": len(users)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)