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


import json
from datetime import datetime
from django.http import JsonResponse
import google.generativeai as genai

genai.configure(api_key="")  # Replace with your key

def get_itinerary(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."})

    destination = request.POST.get("destination")
    checkin = request.POST.get("checkin")
    checkout = request.POST.get("checkout")
    budget = request.POST.get("budget")

    if not all([destination, checkin, checkout, budget]):
        return JsonResponse({"error": "Missing input."})

    try:
        total_days = (datetime.strptime(checkout, "%Y-%m-%d") - datetime.strptime(checkin, "%Y-%m-%d")).days + 1
        if total_days <= 0:
            return JsonResponse({"error": "Invalid date range."})

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are a professional travel planner.
Destination: {destination}
Duration: {total_days} days
Budget: ₹{budget}

Provide ONLY valid JSON output like this:
{{
  "description": "Short intro about {destination}",
  "spots": ["Spot1", "Spot2", ..., "Spot10"],
  "itinerary": [
    "Day 1: Morning - ..., Afternoon - ..., Evening - ...",
    "Day 2: Morning - ..., Afternoon - ..., Evening - ..."
  ]
}}
No markdown, no backticks, no explanations. Only JSON.
"""
        gemini_resp = model.generate_content(prompt)
        text_output = gemini_resp.candidates[0].content.parts[0].text.strip()

        # Try JSON parsing
        try:
            parsed = json.loads(text_output)
        except:
            # Attempt a simple correction: extract first {...} block
            start = text_output.find("{")
            end = text_output.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(text_output[start:end])
                except:
                    parsed = {
                        "description": f"Explore {destination}",
                        "spots": [f"Visit {destination} landmarks"],
                        "itinerary": [f"Day {i+1}: Explore {destination}" for i in range(total_days)]
                    }
            else:
                parsed = {
                    "description": f"Explore {destination}",
                    "spots": [f"Visit {destination} landmarks"],
                    "itinerary": [f"Day {i+1}: Explore {destination}" for i in range(total_days)]
                }

        return JsonResponse({
            "description": parsed.get("description", f"Explore {destination}"),
            "best_time": "Winters (Nov to Feb)",
            "budget": f"₹{budget} for {total_days} days",
            "spots": parsed.get("spots", []),
            "itinerary": parsed.get("itinerary", []),
            "image": "",
        })

    except Exception as e:
        return JsonResponse({"error": f"Gemini error: {str(e)}"})
