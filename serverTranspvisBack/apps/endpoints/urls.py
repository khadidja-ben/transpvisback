from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from apps.endpoints.views import EndpointViewSet
from apps.endpoints.views import MLAlgorithmViewSet
from apps.endpoints.views import MLAlgorithmStatusViewSet
from apps.endpoints.views import MLRequestViewSet
from apps.endpoints.views import PredictView
from apps.endpoints.views import getAlgoView
from apps.endpoints.views import PredictLSTMView
from apps.endpoints.views import PredictClassView
from apps.endpoints.views import PredictionFunctionView

router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    url(r"^api/v2/", include(router.urls)),
    url(r"^api/v2/(?P<endpoint_name>.+)/predict$", PredictView.as_view(), name="predict"),
    url(r"^api/v2/(?P<endpoint_name>.+)/getalgo$", getAlgoView.as_view(), name="predict"),
    url(r"^api/v2/(?P<endpoint_name>.+)/predictLSTM$", PredictLSTMView.as_view(), name="predict"),
    url(r"^api/v2/(?P<endpoint_name>.+)/predictClass$", PredictClassView.as_view(), name="predict"),
    url(r"^api/v2/(?P<endpoint_name>.+)/predictModel$", PredictionFunctionView.as_view(), name="predict"),

]