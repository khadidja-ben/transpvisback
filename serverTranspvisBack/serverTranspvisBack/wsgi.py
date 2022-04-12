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

try:
    registry = MLRegistry() # create ML registry
    # Naive Bayes Classifier
    nv = NaiveBayesClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=nv,
                            algorithm_name="Naive Bayes",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="khadidja",
                            algorithm_description="Naive Bayes with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(NaiveBayesClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))