from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import cv2
import numpy as np
import os
import uuid
from src.detector import DrivingDetector
from src.assessment import DrivingAssessment
from src.config import Config

app = FastAPI()
config = Config()
detector = DrivingDetector(config)
assessor = DrivingAssessment(config)

# Serve static files (for HTML/JS frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process image
        obj_results = detector.detect_objects(image)
        lane_results = detector.detect_lanes(image)
        
        detector_results = {
            'lane_departure': detector.check_lane_departure(image.shape[1] // 2),
            'speed': 75,  # Simulated speed
            'aggressive_turn': False
        }
        
        abnormalities = assessor.assess_behavior(detector_results)
        
        return JSONResponse({
            "status": "success",
            "abnormalities": [ab.__dict__ for ab in abnormalities],
            "objects": obj_results['objects'].tolist() if obj_results['objects'] is not None else [],
            "lanes": lane_results.tolist() if lane_results is not None else []
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h1>Driving Behavior Detection</h1>
            <form action="/api/analyze-image" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <button type="submit">Analyze</button>
            </form>
        </body>
    </html>
    """