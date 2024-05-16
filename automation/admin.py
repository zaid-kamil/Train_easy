from django.contrib import admin
from .models import Dataset,  Preprocessing, AlgorithmSelection, MetricSelection, Training
# Register your models here.
admin.site.register(Dataset)
admin.site.register(Preprocessing)
admin.site.register(AlgorithmSelection)
admin.site.register(MetricSelection)
admin.site.register(Training)
