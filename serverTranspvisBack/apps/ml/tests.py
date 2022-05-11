from django.test import TestCase
from apps.ml.income_classifier.naiveBayes import NaiveBayesClassifier
import pandas as pd

import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data ={
            "paragraph": "effective date march 12 2020"
        } 
        # input_data = pd.DataFrame(input_data, index=[0])
        # print (input_data)
        my_alg = NaiveBayesClassifier()
        response = my_alg.compute_prediction(input_data)
        # self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        # self.assertEqual('>50K', response['label'])

    def test_registry(self):
            registry = MLRegistry()
            self.assertEqual(len(registry.endpoints), 0)
            endpoint_name = "income_classifier"
            algorithm_object = NaiveBayesClassifier()
            algorithm_name = "Naive Bayes"
            algorithm_status = "production"
            algorithm_version = "0.0.1"
            algorithm_owner = "khadidja"
            algorithm_description = "Naive Bayes with simple pre- and post-processing"
            algorithm_code = inspect.getsource(NaiveBayesClassifier)
            # add to registry
            registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                        algorithm_status, algorithm_version, algorithm_owner,
                        algorithm_description, algorithm_code)
            # there should be one endpoint available
            self.assertEqual(len(registry.endpoints), 1)