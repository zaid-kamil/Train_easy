from django.urls import path
from . import views

urlpatterns = [
    path('upload/dataset', views.upload_dataset, name='upload_dataset'),
    path('view/dataset', views.view_dataset, name='view_dataset'),
    path('preprocess/selection', views.preprocess_selection, name='preprocess_selection'),
    path('algorithm/selection', views.algorithm_selection, name='algorithm_selection'),
    path('metric/selection', views.metric_selection, name='metric_selection'),
    path('training', views.training, name='training'),
    path('finalize', views.finalize_pipeline, name='finalize'),
    path('execute', views.execute_pipeline, name='execute'),
    path('my_models', views.my_models, name='my_models'),
    # delete training model
    path('delete/model/<int:pk>', views.delete_model, name='delete_model'),
    # download model
    path('download/model/<int:pk>', views.download_model, name='download_model'),
]

api_urlpatterns = [
    path('api/upload/dataset', views.upload_dataset_api, name='upload_dataset_api'),
    path('api/view/dataset', views.view_dataset_api, name='view_dataset_api'),
]


urlpatterns += api_urlpatterns