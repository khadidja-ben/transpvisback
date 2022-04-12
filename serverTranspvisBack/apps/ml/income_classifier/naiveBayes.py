import joblib
import os
import pandas as pd

class NaiveBayesClassifier:
    # the constructor which loads preprocessing objects and Naive Bayes object (created with Jupyter notebook)
    def __init__(self): 
        CURRENT_DIR = os.path.dirname(__file__)
        TEMPLATE_DIRS = (
            os.path.join(CURRENT_DIR, '../../../machineLearning/')
        )
        self.encoders = joblib.load(TEMPLATE_DIRS + "encoders.joblib")
        self.model = joblib.load(TEMPLATE_DIRS + "naive_bayes.joblib")

    #the method which takes as input JSON data, converts it to Pandas DataFrame and apply pre-processing
    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # # convert categoricals
        # for column in [
        #     "true",
        #     "false",
        # ]:
        #     categorical_convert = self.encoders[column]
        #     input_data[column] = categorical_convert.transform(input_data[column])

        return input_data

    # the method that calls ML for computing predictions on prepared data
    def predict(self, input_data):
        return self.model.predict(input_data)

    # the method that applies post-processing on prediction values
    def postprocessing(self, input_data):
        label = "<=50K"
        if input_data[0]> 0.5:
            label = ">50K"
        return {"probability": input_data[0], "label": label, "status": "OK"}

    # the method that combines: preprocessing, predict and postprocessing and returns JSON object with the response
    def compute_prediction(self, input_data):
        try:
            # input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction