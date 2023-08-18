from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import pandas as pd
import secrets
import numpy as np

app = Flask(__name__)

model_path = 'C:/Users/Arsath khan/Internships/IBM SkillsBuild/frontend/model_LR.pkl'
scaler_path = 'C:/Users/Arsath khan/Internships/IBM SkillsBuild/frontend/scaler_MFI_2.pkl'

with open('model_LR.pkl', 'rb') as f:
    model = pickle.load(f)

with open("scaler_MFI_2.pkl", 'rb') as f:
    scaler = pickle.load(f)
    
# Generate a secret key
secret_key = secrets.token_hex(16)
app.secret_key = secret_key
    

### recommendation
def recommendation(score):
    
    ## acitivity data
    activity_data = {
        'low': {
            'Meditation' : ['Guided Meditation', 'Breathing Exercises'],
            'Mindfulness Exercises': ['Body Scan', 'Walking Meditation'],
            'Self-Care Routine': ['Take a Bath', 'Read a Book'],
            'Relaxation Exercises': ['Progressive Muscle Relaxation', 'Yoga'],
            'Physical Exercises': ['Light Stretching', 'Short Walks'],
            'Creative Expression': ['Drawing', 'Coloring Book'],
            'Gratitude Journaling': ['Write down three things youre grateful for', 'Reflect on positive experiences'],
            'Social Activities': ['Call or meet a friend', 'Join a club or group'],
        },
        'moderate': {
            'Meditation': ['Mindful Sitting', 'Loving-Kindness Meditation'],
            'Mindfulness Exercises': ['Mindful Eating', 'Nature Observation'],
            'Self-Care Routine': ['Practice Gratitude', 'Listen to Music'],
            'Relaxation Exercises': ['Deep Breathing', 'Visualization'],
            'Physical Exercises': ['Yoga Class', 'Dancing'],
            'Gratitude Journaling': ['Write a gratitude letter to someone', 'Create a gratitude collage'],
            'Creative Expression': ['Writing in a journal', 'Playing a musical instrument'],
            'Social Activities': ['Volunteer for a cause you care about', 'Host a small gathering'],
        },
        'high': {
            'Meditation': ['Vipassana Meditation', 'Transcendental Meditation'],
            'Mindfulness Exercises': ['Mindful Walking', 'Mindful Journaling'],
            'Self-Care Routine': ['Connect with Friends', 'Practice a Hobby'],
            'Relaxation Exercises': ['Tai Chi', 'Qi Gong'],
            'Physical Exercises': ['Hiking in nature', 'Group fitness classes'],
            'Gratitude Journaling': ['Practice daily affirmations', 'Mentor someone'],
            'Creative Expression': ['Painting', 'Photography'],
            'Social Activities': ['Organize a community event', 'Lead a workshop'],
        }
    }
    
    
    ### score into classes
    
    mean_score = 666.751503
    std_dev = 45.019868

    if score < (mean_score - std_dev):  ## score < 621.731635
        fitness_class = 'low'
    elif (mean_score - std_dev) <= score <= (mean_score + std_dev):   # 621.731635 <= score <= 711.7713709999999
        fitness_class = 'moderate'
    else:
        fitness_class = 'high'
        
    ## categorize activites
    
    recommended_activities = []
    user_profile = ['Meditation', 'Mindfulness Exercises', 'Self-Care Routine', 'Relaxation Exercises', 
                'Physical Exercises','Gratitude Journaling', 'Creative Expression', 'Social Activities']
    
    for category, activities in activity_data[fitness_class].items():
        if category in user_profile:
            recommended_activities.extend(activities)
            
    return recommended_activities



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Inside predict")
    form_data = request.form
    # print(form_data)
    # Access form data using the correct keys
    FRUITS_VEGGIES = form_data['FRUITS_VEGGIES']
    DAILY_STRESS = form_data['DAILY_STRESS']
    PLACES_VISITED = form_data['PLACES_VISITED']
    SUPPORTING_OTHERS = form_data['SUPPORTING_OTHERS']
    SOCIAL_NETWORK = form_data['SOCIAL_NETWORK']
    ACHIEVEMENT = form_data['ACHIEVEMENT']
    DONATION = form_data['DONATION']
    BMI_RANGE = form_data['BMI_RANGE']
    TODO_COMPLETED = form_data['TODO_COMPLETED']
    DAILY_STEPS = form_data['DAILY_STEPS']
    SLEEP_HOURS = form_data['SLEEP_HOURS']
    DAILY_SHOUTING = form_data['DAILY_SHOUTING']
    SUFFICIENT_INCOME = form_data['SUFFICIENT_INCOME']
    TIME_FOR_PASSION = form_data['TIME_FOR_PASSION']
    WEEKLY_MEDITATION = form_data['WEEKLY_MEDITATION']
    AGE = form_data['AGE']
    GENDER = form_data['GENDER']
    
    #dataframe
    
    ip_df = pd.DataFrame(data=[[FRUITS_VEGGIES,DAILY_STRESS,PLACES_VISITED,SUPPORTING_OTHERS,SOCIAL_NETWORK,
                                ACHIEVEMENT,DONATION,BMI_RANGE,TODO_COMPLETED,DAILY_STEPS,SLEEP_HOURS,
                                DAILY_SHOUTING,SUFFICIENT_INCOME,TIME_FOR_PASSION,WEEKLY_MEDITATION,AGE,GENDER]],
                         columns=['FRUITS_VEGGIES','DAILY_STRESS','PLACES_VISITED','SUPPORTING_OTHERS','SOCIAL_NETWORK',
                                 'ACHIEVEMENT','DONATION','BMI_RANGE','TODO_COMPLETED','DAILY_STEPS','SLEEP_HOURS',
                                 'DAILY_SHOUTING','SUFFICIENT_INCOME','TIME_FOR_PASSION','WEEKLY_MEDITATION','AGE','GENDER'])
    
    #encoding
    ip_df['GENDER'] = ip_df['GENDER'].map({'Male': 1, 'Female': 0})
    
    #scaling
    ip_sc = scaler.transform(ip_df)
    
    # Perform the prediction using the loaded model
    prediction = model.predict(ip_sc)
    op = prediction[0]
    
    # classify the mental fitness
    def classify_mental_fitness(score):
        mean_score = 666.751503
        std_dev = 45.019868

        if score < (mean_score - std_dev):  ## score < 621.731635
            return 'Low'
        elif (mean_score - std_dev) <= score <= (mean_score + std_dev):   # 621.731635 <= score <= 711.7713709999999
            return 'Moderate'
        else:
            return 'High'
        
    # output for classification
    def get_category_details(category):
        if category == 'Low':
            return "Your predicted mental fitness rate is low. It's important to prioritize your mental well-being and seek professional help. Consider consulting with a mental health professional or therapist for guidance and support. Focus on self-care activities, engage in hobbies that bring you joy, and maintain a strong support network."
        elif category == 'Moderate':
            return "Your predicted mental fitness rate is moderate. Continue to pay attention to your mental well-being and consider engaging in activities that promote positive mental health. Explore mindfulness exercises, practice self-reflection, and seek support from loved ones or support groups. Consider consulting with a mental health professional if needed."
        else:
            return "Congratulations! Your predicted mental fitness rate is high. This indicates a positive mental well-being. Continue to prioritize your mental health by engaging in activities that promote well-being, maintaining healthy habits, and nurturing strong relationships. Remember to practice self-care and be mindful of any changes that may occur."

    category = classify_mental_fitness(op)
    details = get_category_details(category)
    
    recommended_activities = recommendation(op)
    session['recommended_activities'] = recommended_activities
    
    # Prepare the response data
    response = {
        'details': details
    }
    
    
    
    return render_template('prediction.html', data=response, round=round)

    

from flask import session

@app.route('/recommendations')
def recommendations_page():
    # Retrieve the recommended activities from the session
    recommended_activities = session.get('recommended_activities', [])
    
    # Clear the session variable to avoid displaying old recommendations on a new request
    session.pop('recommended_activities', None)
    
    image_urls = {
        "Guided Meditation" : "static/medi.jpg",
        "Breathing Exercises" : "static/breath.jpg",
        # "Mindfulness Exercises" : "static/mind.jpg",
        
    }
    
    return render_template("recommendation.html", recommended_activities=recommended_activities, image_urls=image_urls)



if __name__ == '__main__':
    app.run(debug=True)

            
