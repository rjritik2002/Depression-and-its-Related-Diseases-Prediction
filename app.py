import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
app = Flask(__name__)

dep_model = pickle.load(open('dep_model.pkl', 'rb'))
disease_model_1 = pickle.load(open('disease_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def homePage():
    return render_template('index.html')

@app.route('/depression')
def depression():
    return render_template('depression.html')


@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/pages')
def pages():
    return render_template('pages.html')



@app.route('/depression_detect', methods=['GET', 'POST'])
def depression_detect():
    gender = request.form.get('gender')
    age = request.form.get('age')
    race = request.form.get('race')
    education_level = request.form.get('education_level')
    marital_status = request.form.get('marital_status')
    household_size = request.form.get('household_size')
    pregnant = request.form.get('pregnant')
    household_income = request.form.get('household_income')
    asthma = request.form.get('asthma')
    ever_overweight = request.form.get('ever_overweight')
    arthritis = request.form.get('arthritis')
    heart_attack = request.form.get('heart_attack')
    liver_condition = request.form.get('liver_condition')
    weight = request.form.get('weight')
    height = request.form.get('height')
    bmi = request.form.get('bmi')
    pulse = request.form.get('pulse')
    total_cholesterol = request.form.get('total_cholesterol')
    glucose = request.form.get('glucose')
    rbc_count = request.form.get('rbc_count')
    haemoglobin = request.form.get('haemoglobin')
    platelet_count = request.form.get('platelet_count')
    full_time_work = request.form.get('full_time_work')
    work_type = request.form.get('work_type')
    out_of_work = request.form.get('out_of_work')
    trouble_sleeping_history = request.form.get('trouble_sleeping_history')
    sleep_hours = request.form.get('sleep_hours')
    drinks_per_occasion = request.form.get('drinks_per_occasion')
    cant_work = request.form.get('cant_work')
    memory_problems = request.form.get('memory_problems')
    cocaine_use = request.form.get('cocaine_use')
    inject_drugs = request.form.get('inject_drugs')
    current_smoker = request.form.get('current_smoker')

    # input_data = [gender, age, race, education_level, marital_status, household_size, pregnant, household_income, asthma, ever_overweight,
    #               arthritis, heart_attack, liver_condition, weight, height, bmi, pulse, total_cholesterol,
    #               glucose, rbc_count, haemoglobin, platelet_count, full_time_work, work_type, out_of_work,
    #               trouble_sleeping_history, sleep_hours, drinks_per_occasion, cant_work, memory_problems,
    #               cocaine_use, inject_drugs, current_smoker
    #               ]
    # input_data = [0, 20, 3, 1, 2, 2, 3, 3, 0, 0, 0, 1, 1, 61.9,
    #               120, 22.5, 10, 0, 0, 0, 1, 1, 0, 0, 3, 1, 5, 3, 0, 1, 1, 0, 3]
    
    input_data = [0,74,5,5,3,1,0,7,0,0,1,0,0,0,0,0,62,122,100,4.44,12.2,141,0,0,3,1,10.5,1,1,1,0,0,0
]

    input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    depression_prediction = dep_model.predict(input_data_reshaped)

    if depression_prediction[0] == 0:
        message = "Patient is not depressed"
        return render_template('depression.html', message=message)
    else:
        disease_message = "⚠️ Patient is depressed"
        return render_template('disease.html',disease_message = disease_message)
        

    # return render_template('depression.html', message=message)
    # print(input_data)
    # print(ever_overweight)
    # return work_type
    # return input_data


@app.route('/disease_detection', methods=['POST'])
def disease_detection():
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')
    pregnancies = request.form.get('pregnancies')
    glucose = request.form.get('glucose')
    bloodpressure = request.form.get('bloodpressure')
    skin_thickness = request.form.get('skin_thickness')
    insulin = request.form.get('insulin')
    bmi = request.form.get('bmi')
    hypertension = request.form.get('hypertension')
    ever_married = request.form.get('ever_married')
    work_type = request.form.get('work_type')
    residence_type = request.form.get('residence_type')
    smoking_status = request.form.get('smoking_status')

    # disease_data_a = [age, sex, cp, trestbps, chol, fbs, restecg,
    #                 thalach, exang, oldpeak, slope, ca, thal, pregnancies, glucose, bloodpressure, skin_thickness, insulin, bmi, hypertension, ever_married, work_type, residence_type, smoking_status]
    
    disease_data_a = [58,0,0,100,248,0,0,122,0,1,1,0,2,0,0,0,0,0,0,0,0,0,0,0]
    
    disease_data = [float(i) for i in disease_data_a]
    
    # changing the input_data to a numpy array
    disease_data_as_numpy_array = np.asarray(disease_data)

# reshape the np array as we are predicting for one instance
    disease_data_reshaped = disease_data_as_numpy_array.reshape(1,-1)

    disease_prediction = disease_model_1.predict(disease_data_reshaped)
    
    if disease_prediction[0] == 'Brain_Stroke':
        return render_template('disease.html', disease_message="⚠️"+" Patient has Brain Stroke")
    elif disease_prediction[0] == 'Diabetes':
        return render_template('disease.html', disease_message="⚠️"+" Patient is diabetic")
    elif disease_prediction[0] == 'Heart_Disease':
        return render_template('disease.html', disease_message="⚠️"+" Patient has heart disease")
    else:
        return render_template('disease.html', disease_message="✅"+" Patient has no disease")
    
    # return render_template('disease.html',message=disease_prediction)


if __name__ == '__main__':
    app.run(debug=True)
