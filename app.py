from flask import Flask, request, jsonify
import pickle, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
df = pd.read_csv('archive/disease-symptom.csv')
df1 = pd.read_csv('archive/Symptom-severity.csv')
loaded_data = pickle.load(open("model.pkl", "rb"))
m = loaded_data["model"]
label_encoder = loaded_data["decoder"]

@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == "POST" or request.method == "GET":
        input_data = request.get_json()["input_data"]
        input_symptom = input_data.split()

        input_d = np.zeros(133)

        for i in input_symptom:
            symptom_index = df1[df1['Symptom'] == i].index
            input_d[symptom_index] = df1['weight'][symptom_index]

        input_d = input_d.reshape(1, -1)
        
        predictions = m.predict(input_d)

        #find index of disease with the max probability
        predicted_category_index = np.argmax(predictions)

        # Decode the value to find disease (from int to string)
        decoded_value = label_encoder.inverse_transform([predicted_category_index])[0]

        # Return the prediction as JSON response
        return jsonify(prediction=decoded_value, input_data=input_data)
    else:
        # Handle other HTTP methods if necessary
        return jsonify(error="Invalid request method")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
