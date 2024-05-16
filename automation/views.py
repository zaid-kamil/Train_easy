from django.shortcuts import render, redirect
from .forms import DatasetUploadForm, PreprocessingForm, AlgorithmSelectionForm, MetricSelectionForm, TrainingForm , My_ModelsForm
from .models import *
from django.http import JsonResponse
from .preprocessing import preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import time
from django.http import HttpResponse
from joblib import dump, load
import os

@login_required
def upload_dataset(request):
    if request.session.get('current_dataset_id'):
        del request.session['current_dataset_id']
    if request.method == 'POST':
        if request.POST.get('dataset_id'):
            request.session['current_dataset_id'] = request.POST.get('dataset_id')
            return redirect('preprocess_selection')
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()
            messages.success(request, 'Dataset uploaded successfully')
            request.session['current_dataset_id'] = dataset.id
            return redirect('preprocess_selection')
    else:
        form = DatasetUploadForm()
    my_datasets= Dataset.objects.filter(user=request.user).all()
    return render(request, 'upload_dataset.html', {'form': form, 'my_datasets': my_datasets})

@login_required
def view_dataset(request):
    datasets = Dataset.objects.all()
    return render(request, 'view_dataset.html', {'datasets': datasets})

# TO BE TAKEN CARE later
def upload_dataset_api(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()
            messages.success(request, 'Dataset uploaded successfully')
            return redirect('preprocess_selection')
    else:
        form = DatasetUploadForm()
    return render(request, 'components/upload_dataset.html', {'form': form})

def view_dataset_api(request):
    datasets = Dataset.objects.all()
    return render(request, 'components/view_dataset.html', {'datasets': datasets})


# Preprocessing view
def preprocess_selection(request):
    # clear old preprocess id
    if request.session.get('current_preprocess_id'):
        del request.session['current_preprocess_id']
    # load the selected dataset
    did = request.session.get('current_dataset_id')
    df = pd.read_csv(Dataset.objects.get(id=did).file, nrows=5, encoding='latin-1')
    form = PreprocessingForm(initial={'dataset': did})
    if request.method == 'POST':
        form = PreprocessingForm(request.POST)
        if form.is_valid():
            preprocessing = form.save(commit=False)
            preprocessing.user = request.user
            preprocessing.save()
            request.session['current_preprocess_id'] = preprocessing.id
            messages.success(request, 'Preprocessing steps selected successfully')
            return redirect('algorithm_selection')
    missing_values_df= df.isnull().sum().reset_index()
    missing_values_df.columns = ['column', 'missing_values']
    columns_dtype = df.dtypes.reset_index()
    ctx = {
        'form':form, 
        'columns': df.columns.tolist(),
        'df': df.to_html(classes='table table-striped table-bordered table-sm'),
        'missing_values': missing_values_df.to_html(classes='table table-striped table-bordered table-sm'),
        'columns_dtype': columns_dtype.to_html(classes='table table-striped table-bordered table-sm')
    }
    return render(request, 'preprocess_selection.html', ctx)
            
            
def algorithm_selection(request):
    if request.session.get('current_algorithm_id'):
        del request.session['current_algorithm_id']
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    form = AlgorithmSelectionForm(initial={'dataset': did, 'preprocessing': pid})
    if request.method == 'POST':
        form = AlgorithmSelectionForm(request.POST)
        if form.is_valid():
            algorithm = form.save(commit=False)
            algorithm.user = request.user
            algorithm.save()
            request.session['current_algorithm_id'] = algorithm.id
            messages.success(request, 'Algorithm selected successfully')
            return redirect('metric_selection')
    df = pd.read_csv(Dataset.objects.get(id=did).file, nrows=5, encoding='latin-1')
    missing_values_df= df.isnull().sum().reset_index()
    missing_values_df.columns = ['column', 'missing_values']
    columns = df.columns.tolist()    
    columns_dtype = df.dtypes.reset_index().rename(columns={0: 'dtype'})
    preprocesses = Preprocessing.objects.get(id=pid)
    ctx = {
        'form': form,
        'columns': columns,
        'df': df.to_html(classes='table table-striped table-bordered table-sm'),
        'missing_values': missing_values_df.to_html(classes='table table-striped table-bordered table-sm'),
        'columns_dtype': columns_dtype.to_html(classes='table table-striped table-bordered table-sm'),
        'preprocesses': preprocesses
    }
    return render(request, 'algorithm_selection.html', ctx)
    
    
def metric_selection(request):
    if request.session.get('current_metric_id'):
        del request.session['current_metric_id']
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    print(did, pid, aid)
    form = MetricSelectionForm(initial={'dataset': did, 'preprocessing': pid, 'algorithm': aid})
    if request.method == 'POST':
        form = MetricSelectionForm(request.POST)
        if form.is_valid():
            metric = form.save(commit=False)
            metric.user = request.user
            metric.save()
            request.session['current_metric_id'] = metric.id
            messages.success(request, 'Metrics selected successfully')
            return redirect('training')
    df = pd.read_csv(Dataset.objects.get(id=did).file, nrows=5, encoding='latin-1')
    missing_values_df= df.isnull().sum().reset_index()
    missing_values_df.columns = ['column', 'missing_values']
    columns = df.columns.tolist()    
    columns_dtype = df.dtypes.reset_index().rename(columns={0: 'dtype'})
    preprocesses = Preprocessing.objects.get(id=pid)
    algos = AlgorithmSelection.objects.get(id=aid)
    ctx = {
        'form': form,
        'columns': columns,
        'df': df.to_html(classes='table table-striped table-bordered table-sm'),
        'missing_values': missing_values_df.to_html(classes='table table-striped table-bordered table-sm'),
        'columns_dtype': columns_dtype.to_html(classes='table table-striped table-bordered table-sm'),
        'preprocesses': preprocesses,
        'algorithms': algos
    }
    return render(request, 'metric_selection.html', ctx)

def training(request):
    if request.session.get('current_training_id'):
        del request.session['current_training_id']
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    mid = request.session.get('current_metric_id')
    form = TrainingForm(initial={'dataset': did, 'preprocessing': pid, 'algorithm': aid, 'metric': mid})
    if request.method == 'POST':
        form = TrainingForm(request.POST)
        if form.is_valid():
            metrics = MetricSelection.objects.get(id=mid)
            algorithms = AlgorithmSelection.objects.get(id=aid)
            
            training = form.save(commit=False)
            training.user = request.user
            training.metric = metrics
            training.algo = algorithms
            
            training.save()
            request.session['current_training_id'] = training.id
            messages.success(request, 'Training started successfully')
            return redirect('finalize')
    ctx = {'form': form}
    return render(request, 'training.html', ctx)
    
        
        
def finalize_pipeline(request):
    did = request.session.get('current_dataset_id')
    pid = request.session.get('current_preprocess_id')
    aid = request.session.get('current_algorithm_id')
    mid = request.session.get('current_metric_id')
    tid = request.session.get('current_training_id')
    print(did, pid, aid, mid, tid)
    
    try:
        dataset = Dataset.objects.get(id=did)
        df = pd.read_csv(dataset.file, nrows=5, encoding='latin-1')
        columns  = df.columns.tolist()
        preprocesses = Preprocessing.objects.get(id=pid)
        algorithms = AlgorithmSelection.objects.get(id=aid)
        metrics = MetricSelection.objects.get(id=mid)
        training = Training.objects.get(id=tid)
        user = request.user
        ctx = {
            'dataset': dataset,
            'preprocesses': preprocesses,
            'algorithms': algorithms,
            'metrics': metrics,
            'training': training,
            'columns': columns,
            'table': df.to_html(classes='table table-striped table-bordered table-sm')
        }
        return render(request, 'finalize.html', ctx)
    except Exception as e:
        print(e)
        messages.error(request, 'Error in finalizing the pipeline, data is corrupted or in wrong format')
        return redirect('upload_dataset')

@csrf_exempt
def execute_pipeline(request):
    if request.method == 'POST':
        start_time = time.time()
        target = request.POST.get('target')
        did = request.session.get('current_dataset_id')
        pid = request.session.get('current_preprocess_id')
        aid = request.session.get('current_algorithm_id')
        mid = request.session.get('current_metric_id')
        tid = request.session.get('current_training_id')
        print(f'executing: {did}, {pid}, {aid}, {mid}, {tid}')
        dataset = Dataset.objects.get(id=did)
        preprocesses = Preprocessing.objects.get(id=pid)
        algorithms = AlgorithmSelection.objects.get(id=aid)
        metrics = MetricSelection.objects.get(id=mid)
        training = Training.objects.get(id=tid)
        user = request.user
        
        
        # load the dataset
        df = pd.read_csv(dataset.file, nrows=10000)
        # check if target column is textual or numerical
        if df[target].dtype == 'object':
            # encode the target column
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target])
        # task is classification or regression
        if df[target].value_counts().count() <= 10:
            task = 'classification'
        else:
            task = 'regression'
        # y traget column name
        print(f'target: {target}')
        print(f'num of values in target: {df[target].value_counts().count()}')
        X = df.drop(target, axis=1)
        # drop categorical column if num cat greater than 10
        for col in X.columns:
            if X[col].dtype == 'object':
                if X[col].value_counts().count() > 10:
                    X.drop(col, axis=1, inplace=True)
        y = df[target]
        # preprocess the data
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(include='object').columns
        num_steps = []
        cat_steps = []
        if preprocesses.imputation:
            # numerical imputation using SimpleImputer
            num_steps.append(('imputer', SimpleImputer(strategy='mean')))
            cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        if preprocesses.normalization:
            # numerical normalization using StandardScaler
            num_steps.append(('scaler', StandardScaler()))
        if preprocesses.encoding:
            # categorical encoding using OneHotEncoder
            cat_steps.append(('encoder', OrdinalEncoder()))
        if preprocesses.feature_selection:
            # feature selection using SelectPercentile
            num_steps.append(('selector', SelectPercentile(chi2, percentile=50)))
        if preprocesses.pca:
            # PCA
            num_steps.append(('pca', PCA(n_components=3)))
        num_pipeline = Pipeline(num_steps)
        cat_pipeline = Pipeline(cat_steps)
        pipelines = {}
        if task == 'classification':
            print('classification')
            if algorithms.linear:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', LogisticRegression())])
                pipelines['linear'] = pipeline
            if algorithms.decision_Tree:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', DecisionTreeClassifier())])
                pipelines['decision_Tree'] = pipeline
            if algorithms.random_Forest:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', RandomForestClassifier(n_estimators=20))])
                pipelines['random_Forest'] = pipeline
            if algorithms.support_Vector_Machines:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', SVC())])
                pipelines['support_Vector_Machines'] = pipeline
            if algorithms.naive_Bayes:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', GaussianNB())])
                pipelines['naive_Bayes'] = pipeline
            if algorithms.knn:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', KNeighborsClassifier(n_neighbors=5))])
                pipelines['knn'] = pipeline

        else:
            print('regression')
            if algorithms.linear:
                pipeline = Pipeline(steps=[('preprocessor', 
                                            ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', LinearRegression())])
                pipelines['linear'] = pipeline
            if algorithms.decision_Tree:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', DecisionTreeRegressor())])
                pipelines['decision_Tree'] = pipeline
            if algorithms.random_Forest:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', RandomForestRegressor(n_estimators=20))])
                pipelines['random_Forest'] = pipeline
            if algorithms.support_Vector_Machines:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', SVR())])
                pipelines['support_Vector_Machines'] = pipeline
            if algorithms.knn:
                pipeline = Pipeline(steps=[('preprocessor', ColumnTransformer(transformers=[('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])),
                                            ('model', KNeighborsRegressor(n_neighbors=5))])
                pipelines['knn'] = pipeline
            
        # split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training.split, random_state=training.random_state)
        except Exception as e:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
        # train the model
        results = {}
        for model_name, model in pipelines.items():
            # disply model details
            print(f'training model: {model_name}')
            print(model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task == 'classification':
                results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
            else:
                results[model_name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred)
                }
        # visualize the results
        graph = None
        if task == 'classification':
            fig, ax = plt.subplots()
            ax.bar(results.keys(), [r['accuracy'] for r in results.values()])
            ax.set_title('Accuracy of classification models')
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            plt.savefig('assets/output/accuracy.png')
            # save to plot
            os.makedirs('assets/output', exist_ok=True)
            viz = Visualizations(dataset=dataset, user=user, model=training, metric=metrics, algo=algorithms, plot='assets/output/accuracy.png')
            viz.save()
            graph = 'assets/output/accuracy.png'
        if task == 'regression':
            fig, ax = plt.subplots()
            ax.bar(results.keys(), [r['mse'] for r in results.values()])
            ax.set_title('MSE of regression models')
            ax.set_xlabel('Models')
            ax.set_ylabel('MSE')
            plt.savefig('assets/output/mse.png')
            # save to plot
            viz = Visualizations(dataset=dataset, user=user, model=training, metric=metrics, algo=algorithms, plot='assets/output/mse.png')
            viz.save()
            graph = 'assets/output/mse.png'
        # store the best model
        best_model = None
        for model_name, result in results.items():
            if best_model is None:
                best_model = model_name
            else:
                if task == 'classification':
                    if result['accuracy'] > results[best_model]['accuracy']:
                        best_model = model_name
                else:
                    if result['mse'] < results[best_model]['mse']:
                        best_model = model_name
        # save the best model
        os.makedirs('models', exist_ok=True)
        dump(pipelines[best_model], f'models/{best_model}.joblib')
        training.model_path = f'models/{best_model}.joblib'
        training.training_time = time.time() - start_time
        print(results)
        if task == 'classification':
            training.training_accuracy = results[best_model]['accuracy']
            training.testing_accuracy = results[best_model]['accuracy']
        else:
            training.training_accuracy = results[best_model]['mse']
            training.testing_accuracy = results[best_model]['mse']
        training.save()
        graph = graph.replace('assets/', 'static/')
        print(graph)
        return HttpResponse('<div class="text-center">Pipeline executed successfully <br> <img src="/'+graph+'"></div>')
        
    else:
        return JsonResponse({'error': 'Invalid request method'})
    
    
@login_required
def my_models(request):
    # get all my models trained
    print(f'username: {request.user}')
    
    models = Training.objects.filter(user=request.user).all()
    for model in models:
        plot =  Visualizations.objects.filter(model=model).first()  
        if plot:
            model.plot = plot.plot.url.replace('/media/assets', '/static')
            print(model.plot)
        else:
            model.plot = None
            
    ctx = {'models': models}
    return render(request, 'my_models.html', ctx)
    

@login_required
def delete_model(request, pk):
    model = Training.objects.get(id=pk)
    model.delete()
    return redirect('my_models')

@login_required
def download_model(request, pk):
    model = Training.objects.get(id=pk)
    response = HttpResponse(model.model_path, content_type='application/force-download')
    response['Content-Disposition'] = f'attachment; filename={model.model_path}'
    return response