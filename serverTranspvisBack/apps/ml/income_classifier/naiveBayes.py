from tkinter.tix import InputOnly
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

class NaiveBayesClassifier:
    # the constructor which loads preprocessing objects and Naive Bayes object (created with Jupyter notebook)
    def __init__(self): 
        CURRENT_DIR = os.path.dirname(__file__)
        TEMPLATE_DIRS = (
            os.path.join(CURRENT_DIR, '../../../machineLearning/')
        )
        self.model = joblib.load(TEMPLATE_DIRS + "naive_bayes.joblib")
        self.vectorizer_from_train_data = joblib.load(TEMPLATE_DIRS + "count_vectorizer.joblib")

    # the method applies pre-processing
    def preprocessing(self, input_data):
        data = self.vectorizer_from_train_data.transform([word.lower() for word in input_data])
        # JSON to array
        return csr_matrix.toarray(data)

    # the method that calls ML for computing predictions on prepared data
    def predict(self, input_data):
        return self.model.predict(input_data)[0]

    # the method that applies post-processing on prediction values
    def postprocessing(self, input_data):
        return {"label": input_data, "status": "OK"}

    # the method that combines: preprocessing, predict and postprocessing and returns JSON object with the response
    def compute_prediction(self, input_data):
        try:
            print (input_data)
            input = self.preprocessing(input_data)
            print ("inpuuut", input)
            prediction = self.predict(input)
            print (prediction)
            prediction = self.postprocessing(prediction)
            print ("pre-prediction: ", prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction