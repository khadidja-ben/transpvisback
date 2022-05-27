"""
WSGI config for serverTranspvisBack project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.naiveBayes import NaiveBayesClassifier
from apps.ml.income_classifier.textGenerator import TextGenerator
from apps.ml.income_classifier.classifier2 import Classifier2
from apps.ml.income_classifier.ML_Model import MLModel

try:
    registry = MLRegistry() # create ML registry
    # Naive Bayes Classifier
    nv = NaiveBayesClassifier()
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=nv,
                            algorithm_name="Naive Bayes",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="khadidja",
                            algorithm_description="Naive Bayes with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(NaiveBayesClassifier))

    lstm = TextGenerator()
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=lstm,
                            algorithm_name="LSTM",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="khadidja",
                            algorithm_description="LSTM - Long Short Memory",
                            algorithm_code=inspect.getsource(TextGenerator))

    nv2 = Classifier2()
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=nv2,
                            algorithm_name="Naive Bayes",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="khadidja",
                            algorithm_description="Naive Bayes",
                            algorithm_code=inspect.getsource(Classifier2))

    ml = MLModel()
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=ml,
                            algorithm_name="full classification model",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="khadidja",
                            algorithm_description="Information Elements classification",
                            algorithm_code=inspect.getsource(MLModel))
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))