from django.shortcuts import render
from django.conf import settings
import os
import joblib
import pandas as pd
from datetime import datetime

# ---------- Load Model & Encoders ----------
MODEL_DIR = os.path.join(settings.BASE_DIR, 'frontend', 'ml_models')

try:
    knn_model = joblib.load(os.path.join(MODEL_DIR, 'knn_model.pkl'))
    le_section = joblib.load(os.path.join(MODEL_DIR, 'le_section.pkl'))
    le_type = joblib.load(os.path.join(MODEL_DIR, 'le_type.pkl'))
    le_destination = joblib.load(os.path.join(MODEL_DIR, 'le_destination.pkl'))
except Exception as e:
    knn_model = None
    le_section = None
    le_type = None
    le_destination = None
    print(f"[ERROR] Model loading failed: {e}")

# ---------- Static Pages ----------
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

def hotel(request):
    return render(request, 'frontend/hotel.html')

def main_page(request):
    return render(request, 'frontend/main.html')

def rishikesh(request):
    return render(request, 'frontend/rishikesh.html')

def goa(request):
    return render(request, 'frontend/goa.html')

def mumbai(request):
    return render(request, 'frontend/mumbai.html')

def manali(request):
    return render(request, 'frontend/manali.html')

def kerala(request):
    return render(request, 'frontend/kerala.html')

def jammu(request):
    return render(request, 'frontend/jammu.html')
# ---------- Destination Prediction ----------
def destination(request):
    result = None
    error = None

    if request.method == 'POST':
        try:
            price = float(request.POST.get('price'))
            section = request.POST.get('section')
            dest_type = request.POST.get('destination_type')

            section_encoded = le_section.transform([section])[0]
            type_encoded = le_type.transform([dest_type])[0]

            # ✅ Pass only 3 features
            input_data = [[section_encoded, type_encoded , price]]

            prediction = knn_model.predict(input_data)[0]
            result = le_destination.inverse_transform([prediction])[0]

        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render(request, 'frontend/destination.html', {
        'result': result,
        'error': error
    })

