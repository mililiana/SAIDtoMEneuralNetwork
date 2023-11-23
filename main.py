import numpy as np
import pandas as pd
import en_core_web_sm
import csv
from googletrans import Translator
from fuzzywuzzy import fuzz
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from flask import Flask, request
import environ
from time import time

env = environ.Env(
    DEBUG=(bool, False)
)

start = time()

environ.Env.read_env()

translator = Translator()

nlp = en_core_web_sm.load()

label_encoder = LabelEncoder()

label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

model = tf.keras.models.load_model(
    "modeldir"
)

symptoms = []
with open('dataset_for_nlp.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) > 1:
            symptoms.extend([symptom.strip().replace("_", " ") for symptom in row[1:]])

csv_file_path_description = "symptom_Description.csv"
descriptions_data = []
with open(csv_file_path_description, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        descriptions_data.append(row)

csv_file_path_precautions = "symptom_precaution.csv"
precautions_data = []
with open(csv_file_path_precautions, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        precautions_data.append(row)

df1 = pd.read_csv('Symptom-severity.csv')

end = time()
print(end - start)


def get_description(decoded_value, translated):
    description = ""
    for row in descriptions_data:
        if row['Disease'] == decoded_value:
            description = row['Description']
            translated_text = translator.translate(description, src='en', dest='uk')
            description = translated.text.capitalize() + " - " + translated_text.text
            break
    return description


def preprocess_user_text(text):
    text_to_translate = text
    user_input = translator.translate(text_to_translate, src='uk', dest='en')
    translated_text = user_input.text
    user_input_tokens = [token.strip() for token in translated_text.split(' ')]

    found_symptoms = set()
    input_text = " ".join(user_input_tokens)  # Об'єднати слова в один текст

    found_symptoms = set()

    for symptom in symptoms:
        symptom_with_underscore = symptom.replace(" ", "_")
        symptom_with_space = symptom.replace("_", " ")

        if fuzz.partial_ratio(input_text, symptom_with_underscore) == 100:
            found_symptoms.add(symptom_with_underscore)
        elif fuzz.partial_ratio(input_text, symptom_with_space) == 100:
            found_symptoms.add(symptom_with_underscore)

    return found_symptoms


def make_prediction(text):
    input_d = np.zeros(133)
    for i in preprocess_user_text(text):
        symptom_index = df1[df1['Symptom'] == i].index
        input_d[symptom_index] = df1['weight'][symptom_index]

    input_d = input_d.reshape(1, -1)
    # make prediction
    predictions = model.predict(input_d)

    # find index of disease with the max probability
    predicted_category_index = np.argmax(predictions)

    # Decode the value to find disease (from int to string)
    decoded_value = label_encoder.inverse_transform([predicted_category_index])

    translated = translator.translate(decoded_value[0], src='en', dest='uk')

    return decoded_value, translated


def get_precautions(decoded_value, translated):
    precautions = ""

    for row in precautions_data:
        if row['Disease'] == decoded_value:
            Precaution1 = row['Precaution_1']
            Precaution2 = row['Precaution_2']
            Precaution3 = row['Precaution_3']
            Precaution4 = row['Precaution_4']

            precautions += "Рекомендації для '" + translated.text + "'\n"

            translated_text = translator.translate(Precaution1, src='en', dest='uk')
            precautions += translated_text.text.capitalize() + "\n"
            translated_text = translator.translate(Precaution2, src='en', dest='uk')
            precautions += translated_text.text.capitalize() + "\n"
            translated_text = translator.translate(Precaution3, src='en', dest='uk')
            precautions += translated_text.text.capitalize() + "\n"
            translated_text = translator.translate(Precaution4, src='en', dest='uk')
            precautions += translated_text.text.capitalize() + "\n"
            break
    return precautions


# if __name__ == "__main__":
#     text = input()
#     make_prediction(text)


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()
    text = request_data["text"]
    start1 = time()
    (decoded_value, translated) = make_prediction(text)
    end1 = time()
    print("Prediction: ", end1-start1)
    start1 = time()
    precations = get_precautions(decoded_value, translated)
    end1 = time()
    print("Precations: ", end1-start1)
    start1 = time()
    description = get_description(decoded_value, translated)
    end1 = time()
    print("Description: ", end1 - start1)
    result = f"Результати для запиту \"{text}\" \n\n {description}\n{precations}\n Будьте здорові!"
    return {"response": result}


if __name__ == "__main__":
    app.run(debug=False, port=env('PORT') or 5000, host="0.0.0.0")
