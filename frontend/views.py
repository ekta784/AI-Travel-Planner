from django.shortcuts import render
from django.conf import settings
import os
import joblib
from datetime import datetime

# ---------- Load Model & Encoders ----------
MODEL_DIR = os.path.join(settings.BASE_DIR, 'frontend', 'ml_models')

try:
    knn_model = joblib.load(os.path.join(MODEL_DIR, 'knn_model.pkl'))
    le_section = joblib.load(os.path.join(MODEL_DIR, 'le_section.pkl'))
    le_type = joblib.load(os.path.join(MODEL_DIR, 'le_type.pkl'))
    le_destination = joblib.load(os.path.join(MODEL_DIR, 'le_destination.pkl'))
except Exception as e:
    knn_model = le_section = le_type = le_destination = None
    print(f"Model loading failed: {e}")

# ---------- Static Page Views ----------
def index(request):
    return render(request, 'frontend/index.html')

def about(request):
    return render(request, 'frontend/about.html')

def contact(request):
    return render(request, 'frontend/contact.html')

def blog(request):
    return render(request, 'frontend/blog.html')

def blog_single(request):
    return render(request, 'frontend/blog-single.html')

def destination(request):
    return render(request, 'frontend/destination.html')

def hotel(request):
    return render(request, 'frontend/hotel.html')

def main_page(request):
    return render(request, 'frontend/main.html')

def rishikesh(request):
    return render(request, 'frontend/rishikesh.html')

# ---------- ML Destination Prediction ----------
def predict_destination(request):
    if request.method == 'POST':
        try:
            check_in = request.POST.get('check_in')
            check_out = request.POST.get('check_out')
            price = float(request.POST.get('price'))
            section = request.POST.get('section')
            dest_type = request.POST.get('destination_type')

            # Validate inputs
            if not all([check_in, check_out, section, dest_type]):
                raise ValueError("All fields are required.")

            # Calculate trip duration in days
            days = (datetime.strptime(check_out, '%Y-%m-%d') - datetime.strptime(check_in, '%Y-%m-%d')).days
            if days <= 0:
                raise ValueError("Check-out date must be after check-in date.")

            # Encode input
            section_enc = le_section.transform([section])[0]
            type_enc = le_type.transform([dest_type])[0]
            input_data = [[days, price, section_enc, type_enc]]

            # Predict
            prediction = knn_model.predict(input_data)[0]
            destination_name = le_destination.inverse_transform([prediction])[0]

            return render(request, 'frontend/predict_result.html', {'destination': destination_name})

        except Exception as e:
            return render(request, 'frontend/predict_result.html', {'error': f"Prediction failed: {e}"})

    return render(request, 'frontend/main.html')  # Fallback for GET requests
