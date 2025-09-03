import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_hospitals(weightages, input_city, hospital_file="data/all_cities_hospitals_extended_diseases.csv", top_k=5):
    df_hospitals = pd.read_csv(hospital_file)

    disease_cols = ['COVID-19', 'Influenza', 'Viral Fever', 'Malaria', 'Dengue', 'Cough', 'Tuberculosis']

    # Build patient vector
    patient_vector_dict = {disease: weight for disease, _, weight in weightages}
    patient_vector = np.array([patient_vector_dict.get(disease, 0) for disease in disease_cols])

    # Compute similarity
    hospital_vectors = df_hospitals[disease_cols].values
    similarities = cosine_similarity([patient_vector], hospital_vectors)[0]
    df_hospitals['Similarity'] = similarities

    # City distance matrix
    city_distances = {
        'MetroCity':    {'MetroCity': 0,  'RiverTown': 50,  'HillVille': 70,  'GreenBay': 80,  'SunPort': 100},
        'RiverTown':    {'MetroCity': 50, 'RiverTown': 0,   'HillVille': 60,  'GreenBay': 70,  'SunPort': 90},
        'HillVille':    {'MetroCity': 70, 'RiverTown': 60,  'HillVille': 0,   'GreenBay': 40,  'SunPort': 70},
        'GreenBay':     {'MetroCity': 80, 'RiverTown': 70,  'HillVille': 40,  'GreenBay': 0,   'SunPort': 50},
        'SunPort':      {'MetroCity': 100,'RiverTown': 90,  'HillVille': 70,  'GreenBay': 50,  'SunPort': 0}
    }
    alpha = 0.0025

    # Assign hospital cities
    def get_city(row):
        city_scores = {city: row[city] for city in city_distances}
        return max(city_scores, key=city_scores.get)

    df_hospitals['City'] = df_hospitals.apply(get_city, axis=1)
    df_hospitals['Distance_km'] = df_hospitals['City'].apply(lambda c: city_distances[input_city][c])
    df_hospitals['Penalty'] = df_hospitals['Distance_km'] * alpha
    df_hospitals['Final Score'] = df_hospitals['Similarity'] - df_hospitals['Penalty']

    recommended = df_hospitals.sort_values('Final Score', ascending=False)[
        ['Hospital Name', 'City', 'Similarity', 'Distance_km', 'Penalty', 'Final Score']
    ].head(top_k)

    return recommended
