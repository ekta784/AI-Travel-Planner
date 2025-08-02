from django.shortcuts import render
from django.conf import settings
import os
import joblib
import pandas as pd
from datetime import datetime
from django.http import JsonResponse
import requests
from datetime import datetime, timedelta
import math

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

def generate_day_plan(spots, days):
    itinerary = []
    daily_spots = [spots[i::days] for i in range(days)]  # Distribute spots over days
    for i, daily in enumerate(daily_spots, start=1):
        plan = f"Day {i}: Visit " + ", ".join(daily)
        itinerary.append(plan)
    return itinerary

def get_itinerary(request):
    if request.method == "POST":
        destination = request.POST.get("destination")
        checkin = request.POST.get("checkin")
        checkout = request.POST.get("checkout")
        budget = request.POST.get("budget")

        if not all([destination, checkin, checkout, budget]):
            return JsonResponse({"error": "Missing input."}, status=400)

        try:
            # Dates & Duration
            start_date = datetime.strptime(checkin, "%Y-%m-%d")
            end_date = datetime.strptime(checkout, "%Y-%m-%d")
            total_days = (end_date - start_date).days + 1
            if total_days <= 0:
                return JsonResponse({"error": "Invalid date range."}, status=400)

            # Wikipedia Description
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{destination}"
            wiki_data = requests.get(wiki_url).json()
            description = wiki_data.get("extract", "No information available.")
            img_url = wiki_data.get("thumbnail", {}).get("source", "")

            # Dummy Best Time Logic
            best_time = "October to March (cooler weather and festivals)"

            # OpenTripMap Spots
            key = "5ae2e3f221c38a28845f05b66b9c9cd4f465f02a2d459fa779425027"
            geo = requests.get(f"https://api.opentripmap.com/0.1/en/places/geoname?name={destination}&apikey={key}").json()
            lat, lon = geo.get("lat"), geo.get("lon")

            places_url = f"https://api.opentripmap.com/0.1/en/places/radius?radius=10000&lon={lon}&lat={lat}&limit=15&apikey={key}"
            places_resp = requests.get(places_url).json()
            spots = [p["properties"]["name"] for p in places_resp.get("features", []) if p["properties"].get("name")]

            itinerary = generate_day_plan(spots, total_days)

            return JsonResponse({
                "description": description,
                "best_time": best_time,
                "budget": f"₹{budget} for {total_days} days",
                "spots": spots[:10],  # Just top 10
                "itinerary": itinerary,
                "image": img_url,
            })

        except Exception as e:
            return JsonResponse({"error": f"Something went wrong: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)