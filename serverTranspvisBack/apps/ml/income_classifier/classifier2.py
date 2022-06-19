from tkinter.tix import InputOnly
import pickle
import os
from scipy.sparse import csr_matrix

class Classifier2:
    # the constructor which loads preprocessing objects and Naive Bayes object (created with Jupyter notebook)
    def __init__(self): 
        CURRENT_DIR = os.path.dirname(__file__)
        TEMPLATE_DIRS = (
            os.path.join(CURRENT_DIR, '../../../machineLearning/')
        )
        self.model = pickle.load(open(TEMPLATE_DIRS + "Classifier2/naive_bayes_2.pkl", "rb"))
        self.vectorizer_from_train_data = pickle.load(open(TEMPLATE_DIRS + "Classifier2/count_vectorizer_2.pkl", "rb"))

    # the method applies pre-processing
    def preprocessing(self, input_data):
        words = input_data["paragraph"]
        data = self.vectorizer_from_train_data.transform([words])
        # JSON to array
        return csr_matrix.toarray(data)

    # the method that calls ML for computing predictions on prepared data
    def predict(self, input_data):
        return self.model.predict(input_data)

    # the method that applies post-processing on prediction values
    def postprocessing(self, input_data):
        return {"label": input_data, "status": "OK"}

    # the method that combines: preprocessing, predict and postprocessing and returns JSON object with the response
    def compute_prediction(self, input_data):
        try:
            input = self.preprocessing(input_data)
            prediction = self.predict(input)[0]
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return prediction