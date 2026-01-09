AI Travel Planner using Machine Learning and Generative AI
Project Overview

The AI Travel Planner is an intelligent travel recommendation system that helps users plan trips based on their preferences such as budget, travel duration, number of travelers, region, and destination type. The project combines machine learning–based recommendation with generative AI to provide personalized destination suggestions along with detailed travel itineraries.

The system uses a K-Nearest Neighbors (KNN) model to recommend similar travel destinations based on user inputs and integrates the Gemini API to generate human-like, day-wise travel plans.

Key Features

Personalized travel destination recommendations

Similarity-based recommendation using KNN algorithm

Budget-aware and duration-based filtering

Dynamic itinerary generation using Generative AI

User-friendly web interface for travel planning

Real-time recommendations without model retraining

Technologies Used

Machine Learning

K-Nearest Neighbors (KNN) – Recommendation system

Generative AI

Google Gemini API – Itinerary generation

Programming Languages

Python

Frameworks & Libraries

Scikit-learn

Pandas

NumPy

Django (Backend framework)

Frontend

HTML

CSS

JavaScript

Dataset

Custom travel recommendation dataset containing destination details, budget, duration, region, and travel type

Purpose

The goal of this project is to simplify travel planning by providing users with intelligent destination recommendations and detailed itineraries in a single platform. Instead of manually searching for destinations and planning schedules, users receive data-driven recommendations and AI-generated travel plans, saving time and improving decision-making.

Project Workflow

1. Data Collection
Travel destination data including region, destination type, budget per person, total days, and number of travelers is collected and stored in a structured dataset.

2. Data Preprocessing
Categorical features such as region and destination type are encoded, and numerical features are scaled to ensure accurate distance calculation for the KNN model.

3. Model Implementation
A KNN-based recommendation system is implemented to find destinations that are most similar to the user’s travel preferences using distance-based similarity.

4. Destination Recommendation
Based on user input, the model returns the top similar destinations that best match the travel criteria.

5. Itinerary Generation
The selected destination details are passed to the Gemini API, which generates a detailed, day-wise travel itinerary in natural language format.

6. Web Integration
The entire system is integrated into a Django-based web application where users can input their travel preferences and view recommendations and itineraries.

Future Enhancements

Integration of hotel and flight booking APIs

Mobile application version

User login and preference history

Real-time pricing updates

Multi-language itinerary generation

Recommendation based on user feedback and ratings
