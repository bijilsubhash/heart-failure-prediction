import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = request.form.get("age")
    sex = request.form.get("sex")
    chest_pain_type = request.form.get("chest-pain-type")
    resting_bp = request.form.get("resting-bp")
    cholestrol = request.form.get("cholestrol")
    fasting_bs = request.form.get("fasting-bs")
    resting_ecg = request.form.get("resting-ecg")
    max_hr = request.form.get("max-hr")
    exercise_angina = request.form.get("exercise-angina")
    old_peak = request.form.get("old-peak")
    st_slope = request.form.get("st-slope")

    try:
        if sex == 'F':
            sex = 0
        else:
            sex = 1
    
        if chest_pain_type == 'ASY':
            chest_pain_type = 0
        elif chest_pain_type == 'ATA':
            chest_pain_type = 1
        elif chest_pain_type == 'NAP':
            chest_pain_type = 2
        else:
            chest_pain_type = 3
        
        if resting_ecg == 'LVH':
            resting_ecg = 0
        elif resting_ecg == 'Normal':
            resting_ecg = 1
        else:
            resting_ecg = 2
        
        if exercise_angina == 'N':
            exercise_angina = 0
        else:
            exercise_angina = 1
        
        if st_slope == 'Down':
            st_slope = 0
        elif st_slope == 'Flat':
            st_slope = 1
        else:
            st_slope = 2
        features = np.array([age, sex, chest_pain_type, resting_bp, cholestrol, fasting_bs, resting_ecg, max_hr, exercise_angina, old_peak, st_slope]).reshape(1,-1)
        prediction = model.predict(features)
    except:
        return render_template('index.html', prediction_text='Input queries cannot be left blank')

    if prediction == 0:
        return render_template('index.html', prediction_text='Heart is functioning normally')
    else:
        return render_template('index.html', prediction_text='High likelihood of heart disease')

if __name__ == "__main__":
    app.run(debug=True)