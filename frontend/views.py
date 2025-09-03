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
import os
import json
import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from sklearn.neighbors import NearestNeighbors   # <-- Import added

# Define models directory inside project
MODEL_DIR = os.path.join(settings.BASE_DIR, "ml_models")

# Load objects
model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))   # base NearestNeighbors model
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
df = joblib.load(os.path.join(MODEL_DIR, "travel_dataset.pkl"))  # dataset for lookup

@csrf_exempt
def predict_destination(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Extract user input
            region = data.get("section")   # frontend sends "section"
            destination_type = data.get("destination_type")
            total_days = int(data.get("total_days"))
            total_persons = int(data.get("total_persons"))
            budget = int(data.get("budget"))

            # Encode categorical inputs
            region_encoded = label_encoders["Region"].transform([region])[0]

            # Filter dataset by destination_type
            df_filtered = df[df["Destination_Type"] == destination_type].copy()
            if df_filtered.empty:
                return JsonResponse({"error": f"No destinations found for type: {destination_type}"}, status=404)

            # Features for filtered dataset
            X_filt = df_filtered[["Region_encoded", "Total_Days", "Total_Persons", "Budget_Per_Person"]]
            X_filt_scaled = scaler.transform(X_filt)

            # Train temporary NN model for this subset
            knn_temp = NearestNeighbors(n_neighbors=min(5, len(df_filtered)), metric="euclidean")
            knn_temp.fit(X_filt_scaled)

            # Encode + scale user input
            user_input = pd.DataFrame(
                [[region_encoded, total_days, total_persons, budget]],
                columns=["Region_encoded", "Total_Days", "Total_Persons", "Budget_Per_Person"]
            )
            user_scaled = scaler.transform(user_input)

            # Get nearest neighbors
            distances, indices = knn_temp.kneighbors(user_scaled, n_neighbors=min(5, len(df_filtered)))

            recommendations = df_filtered.iloc[indices[0]][
                ["Destination_Name", "Destination_Type", "Region", "Total_Days", "Total_Persons", "Budget_Per_Person"]
            ]

            return JsonResponse({"recommendations": recommendations.to_dict(orient="records")})
        
        except Exception as e:
            import traceback
            print("🔥 ERROR in predict_destination:", e)
            traceback.print_exc()   # <-- shows full stack trace in console
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)



# ---------- Static Pages ----------
def index(request):
    return render(request, 'frontend/index.html')

def about(request):
    return render(request, 'frontend/about.html')

def destination(request):
    return render(request, 'frontend/destination.html')

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


def generate_day_plan(spots, days):
    itinerary = []
    spots_per_day = [spots[i::days] for i in range(days)]  # Spread spots across days

    for i, daily_spots in enumerate(spots_per_day, start=1):
        morning = daily_spots[:1]
        afternoon = daily_spots[1:2]
        evening = daily_spots[2:3]

        day_plan = f"""
        <strong>Day {i}</strong><br>
        🕗 8:00 AM - Breakfast at local cafe<br>
        🕘 9:30 AM - Visit: {morning[0] if morning else 'Relax at hotel'}<br>
        🕛 12:30 PM - Lunch break<br>
        🕐 2:00 PM - Visit: {afternoon[0] if afternoon else 'Explore nearby markets'}<br>
        🕔 5:00 PM - Tea & Relax<br>
        🕕 6:30 PM - Visit: {evening[0] if evening else 'Evening walk or leisure'}<br>
        🕗 8:00 PM - Dinner at recommended restaurant
        """
        itinerary.append(day_plan.strip())

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
            try:
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{destination}"
                wiki_data = requests.get(wiki_url, timeout=5).json()
                description = wiki_data.get("extract", "No information available.")
                img_url = wiki_data.get("thumbnail", {}).get("source", "")
            except Exception:
                description, img_url = "No information available.", ""

            # Weather API (Safe)
            best_time = "Weather data unavailable."
            try:
                if lat and lon:
                    weather_resp = requests.get(
                        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true",
                        timeout=5
                    ).json()
                    if "current_weather" in weather_resp:
                        temp = weather_resp["current_weather"].get("temperature", "N/A")
                        wind = weather_resp["current_weather"].get("windspeed", "N/A")
                        best_time = f"Current temperature: {temp}°C, Windspeed: {wind} km/h"
            except Exception:
                pass

            # OpenTripMap API (Safe)
            spots = []
            try:
                key = "5ae2e3f221c38a28845f05b66b9c9cd4f465f02a2d459fa779425027"
                geo = requests.get(
                    f"https://api.opentripmap.com/0.1/en/places/geoname?name={destination}&apikey={key}",
                    timeout=5
                ).json()
                lat, lon = geo.get("lat"), geo.get("lon")
                if lat and lon:
                    places_url = f"https://api.opentripmap.com/0.1/en/places/radius?radius=10000&lon={lon}&lat={lat}&limit=15&apikey={key}"
                    places_resp = requests.get(places_url, timeout=5).json()
                    spots = [p["properties"]["name"] for p in places_resp.get("features", []) if p["properties"].get("name")]
            except Exception:
                spots = ["Explore local culture", "Visit city center"]

            itinerary = generate_day_plan(spots, total_days)

            return JsonResponse({
                "description": description,
                "best_time": best_time,
                "budget": f"₹{budget} for {total_days} days",
                "spots": spots[:10],
                "itinerary": itinerary,
                "image": img_url,
            })

        except Exception as e:
            return JsonResponse({"error": f"Something went wrong: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)
