"""
Olive Disease Detection - FastAPI Production Server
High-performance inference with health checks and Arabic/English recommendations
"""

import os
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import logging
from typing import Dict, List, Optional
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """Handles image loading, resizing, and normalization for inference"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process_image(self, image_file: UploadFile) -> torch.Tensor:
        """
        Process uploaded image file to tensor.
        
        Args:
            image_file: FastAPI UploadFile object
            
        Returns:
            Preprocessed image tensor (1, 3, 224, 224)
            
        Raises:
            ValueError: If image format is invalid
        """
        try:
            # Read image bytes
            image_data = image_file.file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            return tensor.unsqueeze(0)
        
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class OliveModel(nn.Module):
    """Transfer Learning Model for Olive Disease Detection"""
    
    def __init__(self, num_classes: int = 4, architecture: str = "MobileNetV2"):
        super(OliveModel, self).__init__()
        
        if architecture == "MobileNetV2":
            self.backbone = models.mobilenet_v2(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.feature_extractor = self.backbone.features
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            
        elif architecture == "EfficientNet-B0":
            self.backbone = models.efficientnet_b0(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.feature_extractor = self.backbone.features
            self.avg_pool = self.backbone.avgpool
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


# ============================================================================
# RECOMMENDATIONS
# ============================================================================

class RecommendationEngine:
    """Generate Arabic and English recommendations based on disease"""
    
    RECOMMENDATIONS = {
        "Healthy": {
            "en": {
                "title": "Healthy Leaf",
                "recommendations": [
                    "Continue regular monitoring",
                    "Maintain current irrigation schedule",
                    "Apply preventive fungicide spray every 3 months",
                    "Ensure proper canopy ventilation"
                ],
                "severity": "Low",
                "action_required": False
            },
            "ar": {
                "title": "ورقة صحية",
                "recommendations": [
                    "استمرار المراقبة المنتظمة",
                    "الحفاظ على جدول الري الحالي",
                    "تطبيق رش فطري وقائي كل 3 أشهر",
                    "ضمان تهوية التاج الجيدة"
                ],
                "severity": "منخفضة",
                "action_required": False
            }
        },
        "Peacock_Spot": {
            "en": {
                "title": "Peacock Spot (Fungal)",
                "recommendations": [
                    "Remove infected leaves immediately",
                    "Apply copper-based fungicide (Bordeaux mixture recommended)",
                    "Improve drainage and reduce leaf wetness",
                    "Increase spray frequency to every 10-14 days",
                    "Prune affected branches to improve air circulation"
                ],
                "severity": "High",
                "action_required": True
            },
            "ar": {
                "title": "بقعة الطاووس (فطرية)",
                "recommendations": [
                    "إزالة الأوراق المصابة فورًا",
                    "تطبيق مبيد فطري نحاسي (خليط بوردو موصى به)",
                    "تحسين الصرف وتقليل رطوبة الأوراق",
                    "زيادة تكرار الرش كل 10-14 يوم",
                    "تقليم الفروع المصابة لتحسين دوران الهواء"
                ],
                "severity": "عالية",
                "action_required": True
            }
        },
        "Olive_Knot": {
            "en": {
                "title": "Olive Knot (Bacterial)",
                "recommendations": [
                    "Prune infected branches 30cm below the knot",
                    "Disinfect pruning tools between cuts (70% ethanol)",
                    "Apply copper hydroxide paste to cut surfaces",
                    "Avoid wounding the tree",
                    "Monitor closely for disease spread"
                ],
                "severity": "Critical",
                "action_required": True
            },
            "ar": {
                "title": "عقدة الزيتون (بكتيرية)",
                "recommendations": [
                    "تقليم الفروع المصابة 30 سم أسفل العقدة",
                    "تطهير أدوات التقليم بين القطعات (كحول 70%)",
                    "تطبيق معجون هيدروكسيد النحاس على سطح القطع",
                    "تجنب جرح الشجرة",
                    "مراقبة عن كثب لانتشار المرض"
                ],
                "severity": "حرجة",
                "action_required": True
            }
        }
    }
    
    @staticmethod
    def get_recommendations(disease_class: str, confidence: float) -> Dict:
        """Get bilingual recommendations for detected disease"""
        if disease_class not in RecommendationEngine.RECOMMENDATIONS:
            return {
                "en": {"title": "Unknown", "recommendations": [], "severity": "Unknown"},
                "ar": {"title": "غير معروف", "recommendations": [], "severity": "غير معروف"}
            }
        
        recommendations = RecommendationEngine.RECOMMENDATIONS[disease_class]
        
        # Adjust recommendations based on confidence
        if confidence < 0.6:
            recommendations["en"]["title"] += " (Low Confidence - Manual Review Recommended)"
            recommendations["ar"]["title"] += " (ثقة منخفضة - المراجعة اليدوية مستحسنة)"
        
        return recommendations


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    timestamp: str
    image_filename: str
    prediction: str
    confidence: float
    all_predictions: Dict[str, float]
    recommendations: Dict
    model_version: str = "1.0"


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    model_loaded: bool
    device: str


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Olive Disease Detection API",
    description="Production-grade API for olive leaf disease classification",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
image_processor = None
class_names = ["Healthy", "Olive_Knot", "Peacock_Spot"]


# ============================================================================
# INITIALIZATION
# ============================================================================

def load_model():
    """Load pre-trained model from disk"""
    global model, image_processor
    
    try:
        # Get model path from environment or default
        model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        # Initialize model architecture
        model = OliveModel(num_classes=len(class_names), architecture="MobileNetV2")
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Initialize image processor
        image_processor = ImageProcessor(image_size=224)
        
        logger.info(f"Model loaded successfully on device: {device}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup"""
    logger.info("Starting Olive Disease Detection API")
    load_model()
    logger.info("API ready for predictions")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for Kubernetes liveness/readiness probes.
    
    Returns:
        HealthResponse: Status and model information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        device=str(device)
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint for monitoring.
    
    Returns:
        JSON with API metrics
    """
    return {
        "status": "operational",
        "model_loaded": model is not None,
        "device": str(device),
        "class_names": class_names,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict disease class for uploaded olive leaf image.
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        PredictionResponse: Prediction, confidence, and recommendations
        
    Raises:
        HTTPException: If image is invalid or model fails
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not image_processor:
        raise HTTPException(status_code=503, detail="Image processor not initialized")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected JPEG or PNG."
        )
    
    try:
        # Process image
        logger.info(f"Processing image: {file.filename}")
        image_tensor = image_processor.process_image(file)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        # Get predictions
        pred_class_idx = probabilities.argmax().item()
        pred_class = class_names[pred_class_idx]
        confidence = float(probabilities[pred_class_idx].item())
        
        # Create all predictions dict
        all_predictions = {
            class_names[i]: float(probabilities[i].item())
            for i in range(len(class_names))
        }
        
        # Get recommendations
        recommendations = RecommendationEngine.get_recommendations(pred_class, confidence)
        
        logger.info(f"Prediction: {pred_class} ({confidence:.2%})")
        
        return PredictionResponse(
            timestamp=datetime.utcnow().isoformat(),
            image_filename=file.filename,
            prediction=pred_class,
            confidence=confidence,
            all_predictions=all_predictions,
            recommendations=recommendations
        )
    
    except ValueError as ve:
        logger.error(f"Image processing error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict disease for multiple images in a single request.
    
    Args:
        files: List of image files
        
    Returns:
        List of predictions for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            # Process image
            image_tensor = image_processor.process_image(file)
            image_tensor = image_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = F.softmax(output, dim=1)[0]
            
            # Get predictions
            pred_class_idx = probabilities.argmax().item()
            pred_class = class_names[pred_class_idx]
            confidence = float(probabilities[pred_class_idx].item())
            
            all_predictions = {
                class_names[i]: float(probabilities[i].item())
                for i in range(len(class_names))
            }
            
            recommendations = RecommendationEngine.get_recommendations(pred_class, confidence)
            
            results.append({
                "filename": file.filename,
                "prediction": pred_class,
                "confidence": confidence,
                "all_predictions": all_predictions,
                "recommendations": recommendations
            })
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_images": len(files),
        "successful": len([r for r in results if "error" not in r]),
        "results": results
    }


@app.get("/info", tags=["Information"])
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_name": "Olive Disease Detection",
        "version": "1.0",
        "architecture": "MobileNetV2 (Transfer Learning)",
        "input_size": 224,
        "classes": class_names,
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API documentation"""
    return {
        "service": "Olive Disease Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch-predict (POST)",
            "info": "/info",
            "metrics": "/metrics",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    workers = int(os.getenv("WORKERS", "4"))
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )