from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import re

app = Flask(__name__)
app.secret_key = 'Raju'
# Load the dataset
dataset = pd.read_csv("Crop and fertilizer dataset.csv")



# Prepare the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(dataset[['District_Name', 'Soil_color']])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, dataset['Crop'], test_size=0.2, random_state=42)
model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
model_crop.fit(X_train, y_train)

def extract_youtube_id(url):
    """
    Extracts the YouTube ID from a URL
    """
    # This regex covers various YouTube URL formats, including short URLs and embed URLs
    regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\s*[^\/\n\s]+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None  # Return None if no ID is found



@app.route('/')
def index():
    # Extract unique values from the dataset for each form field
    districts = dataset['District_Name'].unique().tolist()
    soil_colors = dataset['Soil_color'].unique().tolist()
    nitrogen_levels = dataset['Nitrogen'].unique().tolist()
    phosphorus_levels = dataset['Phosphorus'].unique().tolist()
    potassium_levels = dataset['Potassium'].unique().tolist()
    ph_levels = dataset['pH'].unique().tolist()
    rainfall_levels = dataset['Rainfall'].unique().tolist()
    temperature_levels = dataset['Temperature'].unique().tolist()

    # Sort the lists if necessary, or convert data types if required
    # Example: nitrogen_levels.sort()

    return render_template('index.html',
                           districts=districts,
                           soil_colors=soil_colors,
                           nitrogen_levels=nitrogen_levels,
                           phosphorus_levels=phosphorus_levels,
                           potassium_levels=potassium_levels,
                           ph_levels=ph_levels,
                           rainfall_levels=rainfall_levels,
                           temperature_levels=temperature_levels)


@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract form data
    district = request.form['district']
    soil_color = request.form['soil_color']
    nitrogen = request.form['nitrogen']
    phosphorus = request.form['phosphorus']
    potassium = request.form['potassium']
    ph = request.form['ph']
    rainfall = request.form['rainfall']
    temperature = request.form['temperature']

    # Example processing
    input_data = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, ph, rainfall, temperature, district, soil_color]],
        columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'District_Name', 'Soil_color']
    )
    input_data_encoded = encoder.transform(input_data[['District_Name', 'Soil_color']])
    predicted_crop = model_crop.predict(input_data_encoded)[0]
    recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop]['Fertilizer'].values[0]
    link = dataset[(dataset['Crop'] == predicted_crop) & (dataset['Fertilizer'] == recommended_fertilizer)]['Link'].values[0]

    session['results'] = {
        'predicted_crop': predicted_crop,
        'recommended_fertilizer': recommended_fertilizer,
        'link': link # Store only the YouTube ID
    }
    return redirect(url_for('results'))   


@app.route('/results')
def results():
    # Retrieve results from session
    results = session.get('results', {})
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
