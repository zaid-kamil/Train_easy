from django import forms
from .models import Dataset, Preprocessing, AlgorithmSelection, MetricSelection, Training, My_Models

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter Project Name'}),
        }
        
class PreprocessingForm(forms.ModelForm):
    class Meta:
        model = Preprocessing
        fields = ['dataset','normalization', 'encoding', 'imputation', 'feature_selection', 'pca']
        
class AlgorithmSelectionForm(forms.ModelForm):
    class Meta:
        model = AlgorithmSelection
        fields = ['dataset','linear', 'decision_Tree', 'random_Forest', 'support_Vector_Machines', 'naive_Bayes', 'knn']
        
class MetricSelectionForm(forms.ModelForm): 
    class Meta:
        model = MetricSelection
        fields = ['dataset','accuracy', 'mse', 'rmse', 'mae', 'confusion_matrix', 'roc_auc', 'precision', 'recall', 'f1']
        
class TrainingForm(forms.ModelForm):
    class Meta:
        model = Training
        fields = ['dataset','split']
        widgets = {
            'split': forms.TextInput(attrs={'placeholder': 'Enter Split Value', 'min': '0', 'max': '1', 'step': '0.01', 'type': 'number', 'required': 'required'}),
        }
        
class My_ModelsForm(forms.ModelForm):
    class Meta:
        model = My_Models
        fields = ['dataset', 'model_name', 'model_type', 'model_accuracy', 'model_mse', 'model_rmse', 'model_mae', 'model_confusion_matrix', 'model_roc_auc', 'model_precision', 'model_recall', 'model_f1']
