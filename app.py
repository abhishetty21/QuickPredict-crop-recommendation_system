from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle as pi

app = Flask(__name__)

# Load models
models = {}
algo = ['Naive Bayes', 'Decision Tree', 'Random Forest']
for name in algo:
    fn = name + ".prediction"
    md = pi.load(open(fn, 'rb'))
    models[name] = md

# Render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Convert form data to a list of values in the order of features
    input_values = [float(data[feat]) for feat in features]
    
    # Create a DataFrame with a single row using the input values
    arr = pd.DataFrame([input_values], columns=features)
    
    results = {}
    for name, model in models.items():
        pred = model.predict(arr)
        results[name] = pred[0]
    print(results['Random Forest'])
    return jsonify({'prediction': results['Random Forest']})

if __name__ == '__main__':
    app.run(debug=True)
