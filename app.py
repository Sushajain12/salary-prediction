from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

# Load the trained model
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('lb_salary.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    experience = float(request.form['experience'])
    gender = request.form['gender']
    gender_encoded = 0 if gender == 'Male' else 1

    features = np.array([[age, experience, gender_encoded]])
    prediction = model.predict(features)
    salary = round(prediction[0], 2)

    return render_template('result.html', salary=salary)

if __name__ == '__main__':
    app.run(debug=True)
