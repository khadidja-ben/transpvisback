import json
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response
# from apps.ml.registry import MLRegistry
from serverTranspvisBack.wsgi import registry

from rest_framework import viewsets
from rest_framework import mixins

from django.shortcuts import render
# from apps.ml.income_classifier.textGenerator import TextGenerator
from django.http import JsonResponse

# from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

from apps.endpoints.models import MLAlgorithm
from apps.endpoints.serializers import MLAlgorithmSerializer

from apps.endpoints.models import MLAlgorithmStatus
from apps.endpoints.serializers import MLAlgorithmStatusSerializer

from apps.endpoints.models import MLRequest
from apps.endpoints.serializers import MLRequestSerializer

class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm = instance.parent_mlalgorithm,
                                                        created_at__lt=instance.created_at,
                                                        active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])

class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin
):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()
    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)



        except Exception as e:
            raise APIException(str(e))

class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()


class getAlgoView(views.APIView):
    def post(self, endpoint_name):
        print("----------------------------------------", )
        algs = MLAlgorithm.objects.filter(name = endpoint_name,status__active=True)
        alg_index = 1
        # algorithm_object = registry.endpoints[algs[alg_index].id]
        return algs


class PredictView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algs = MLAlgorithm.objects.filter(parent_endpoint__name = endpoint_name)
        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.compute_prediction(request.data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id
        prediction["data"] = request.data

        return Response(prediction)

class PredictLSTMView(views.APIView):

    def post(self, request, endpoint_name, format=None):

        algs = MLAlgorithm.objects.filter(parent_endpoint__name = endpoint_name)
        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 1
        algorithm_object = registry.endpoints[2]
        prediction = algorithm_object.compute_prediction(request.data)
        summary = prediction["summary"] if "summary" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=summary,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()
        prediction["request_id"] = ml_request.id
        prediction["data"] = request.data

        return Response(prediction)

class PredictClassView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algs = MLAlgorithm.objects.filter(parent_endpoint__name = endpoint_name)
        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        algorithm_object = registry.endpoints[3]
        prediction = algorithm_object.compute_prediction(request.data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id
        prediction["data"] = request.data

        return Response(prediction)

class PredictionFunctionView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algs = MLAlgorithm.objects.filter(parent_endpoint__name = endpoint_name)
        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        algorithm_object = registry.endpoints[4]
        data = request.data["paragraph"]
        print (data)
        prediction = algorithm_object.mlmodel(data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id
        # prediction["data"] = request.data

        return Response(prediction)


class paragraphs(views.APIView):
    def post (self, request, endpoint_name, format=None):
        data = request.data["paragraph"]
        data= data.split("\n")
        # print ("data :::::::::::::::::", data)
        print("Number of paragraphs is : ", len(data))
        # delete empty lines from the document 
        table = []
        for index, p in enumerate(data): 
            if data[index] == "":
                del data[index]
            else: 
                table.append(data[index])
            
        print("Number of paragraphs is : ", len(data))
        # print ("table", table)
        # prediction["data"] = request.data
        return Response(table)

class PredictionFunctionViewFull(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algs = MLAlgorithm.objects.filter(parent_endpoint__name = endpoint_name)
        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        algorithm_object = registry.endpoints[4]
        data = request.data
        print("data = ",data)
        data= data.split("\n")
        table = []
        for index, p in enumerate(data): 
            if data[index] == "":
                del data[index]
            else: 
                table.append(data[index])
        # print(table)        
        predictionResults = []
        for x in table:
            print ("xxxxxx = ",x)
            prediction = algorithm_object.mlmodel(x)
            label = prediction["label"] if "label" in prediction else "error"
            # if information element already exists no need to add it !
            ml_request = MLRequest(
                input_data=json.dumps(x),
                full_response=prediction,
                response=label,
                feedback="",
                parent_mlalgorithm=algs[alg_index],
            )
            ml_request.save()
            prediction["request_id"] = ml_request.id
            if (prediction["summary"] not in predictionResults):
                predictionResults.append(prediction)
                print (prediction ["summary"])

        # predictionResults.append({'summary': 'personal information', 'label': 'data'})
        # predictionResults.append({'summary': 'payment information', 'label': 'data'})
        # predictionResults.append({'summary': 'cookies information', 'label': 'policy'})
        # predictionResults.append({'summary': 'provide services', 'label': 'policy'})
        # predictionResults.append({'summary': 'assuring security', 'label':'process'})
        # predictionResults.append({'summary': 'personalize experience', 'label':'process'})

        print(predictionResults)
# return {"paragraph":input_data,"summary": text['summary'], "label": classification['label'], "status": "OK"}

        return Response(predictionResults)

        # prediction = algorithm_object.mlmodel(data)

        # label = prediction["label"] if "label" in prediction else "error"
        # ml_request = MLRequest(
        #     input_data=json.dumps(request.data),
        #     full_response=prediction,
        #     response=label,
        #     feedback="",
        #     parent_mlalgorithm=algs[alg_index],
        # )
        # ml_request.save()

        # prediction["request_id"] = ml_request.id
        # prediction["data"] = request.data

