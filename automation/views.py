from django.shortcuts import render, redirect
from .forms import DatasetUploadForm, PreprocessingForm, AlgorithmSelectionForm, MetricSelectionForm, TrainingForm , My_ModelsForm
from .models import *
from django.http import JsonResponse
from .preprocessing import preprocess_data
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import time
from django.http import HttpResponse
from joblib import dump, load
import os
from sklearn.utils import estimator_html_repr

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
    trainings = Training.objects.filter(user=request.user).all()
    return render(request, 'upload_dataset.html', {'form': form, 'my_datasets': my_datasets,'trainings': trainings})

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
    form = TrainingForm(initial={'dataset': did, 
                                 'preprocessing': pid, 
                                 'algorithm': aid,
                                 'metric': mid}
                        )
    if request.method == 'POST':
        form = TrainingForm(request.POST)
        if form.is_valid():
            metrics = MetricSelection.objects.get(id=mid)
            algorithms = AlgorithmSelection.objects.get(id=aid)
            preprocesses = Preprocessing.objects.get(id=pid)
            training = form.save(commit=False)
            training.user = request.user
            training.metric = metrics
            training.algo = algorithms
            training.preprocessing = preprocesses
            training.save()
            request.session['current_training_id'] = training.id
            messages.success(request, 'Training started successfully')
            return redirect('finalize')
    ctx = {'form': form}
    return render(request, 'training.html', ctx)
   
   
def select_training(request, pk):
    training = Training.objects.get(id=pk)
    request.session['current_dataset_id'] = training.dataset.id
    request.session['current_preprocess_id'] = training.preprocessing.id
    request.session['current_algorithm_id'] = training.algo.id
    request.session['current_metric_id'] = training.metric.id
    request.session['current_training_id'] = training.id
    return redirect('finalize') 
        
        
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
    

# regression pipeline
def regression_pipeline(request,df, did, pid, aid, mid, tid, target):
    pptask = Preprocessing.objects.get(id=pid)  
    algos = AlgorithmSelection.objects.get(id=aid)
    metrics = MetricSelection.objects.get(id=mid)
    training = Training.objects.get(id=tid)
    # check if normalization is selected
    
    # create a blank pipeline
    all_models = []
    X, y = df.drop(target, axis=1), df[target]
    num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_trans_step = []
    cat_trans_step = []
    if pptask.imputation:
        if num_cols:
            num_trans_step.append(('imputer', SimpleImputer(strategy='mean')))
        if cat_cols:
            cat_trans_step.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
    if pptask.normalization:
        if num_cols:
            num_trans_step.append(('scaler', StandardScaler()))
    
    if pptask.encoding:
        if cat_cols:
            cat_trans_step.append(('encoder', OrdinalEncoder()))
    numeric_transformer, categorical_transformer = None, None
    if num_trans_step:       
        numeric_transformer = Pipeline(steps=num_trans_step)
    if cat_trans_step:
        categorical_transformer = Pipeline(steps=cat_trans_step)
    
    if numeric_transformer is not None and categorical_transformer is not None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )
    elif numeric_transformer:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols)
            ]
        )
    elif categorical_transformer:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, cat_cols)
            ]
        )
    else:
        preprocessor = None
    
    if algos.linear:
        steps = [('preprocessor', preprocessor)]
        if pptask.feature_selection:
            steps.append(('feature_selection', SelectPercentile(chi2)))
        if pptask.pca and df.shape[1] > 3:
            steps.append(('pca', PCA(n_components=3)))
        linear = Pipeline(steps=steps + [('classifier', ElasticNetCV())])
        all_models.append(('linear', linear))
    
    if algos.decision_Tree:
        steps = [('preprocessor', preprocessor)]
        if pptask.feature_selection:
            steps.append(('feature_selection', SelectPercentile(chi2)))
        if pptask.pca and df.shape[1] > 3:
            steps.append(('pca', PCA(n_components=3)))
        decision_tree = Pipeline(steps=steps + [('classifier', DecisionTreeRegressor())])
        all_models.append(('decision_tree', decision_tree))
    
    if algos.random_Forest:
        steps = [('preprocessor', preprocessor)]
        if pptask.feature_selection:
            steps.append(('feature_selection', SelectPercentile(chi2)))
        if pptask.pca and df.shape[1] > 3:
            steps.append(('pca', PCA(n_components=3)))
        random_forest = Pipeline(steps=steps + [('classifier', RandomForestRegressor())])
        all_models.append(('random_forest', random_forest))
        
    if algos.support_Vector_Machines:
        steps = [('preprocessor', preprocessor)]
        if pptask.feature_selection:
            steps.append(('feature_selection', SelectPercentile(chi2)))
        if pptask.pca and df.shape[1] > 3:
            steps.append(('pca', PCA(n_components=3)))
        svm = Pipeline(steps=steps + [('classifier', SVR())])
        all_models.append(('svm', svm))
        
        
    if algos.knn:
        steps = [('preprocessor', preprocessor)]
        if pptask.feature_selection:
            steps.append(('feature_selection', SelectPercentile(chi2)))
        if pptask.pca and df.shape[1] > 3:
            steps.append(('pca', PCA(n_components=3)))
        knn = Pipeline(steps=steps + [('classifier', KNeighborsRegressor())])
        all_models.append(('knn', knn))
        
    
    # save the pipeline diagram to visualize
    best_model = None
    best_results = None
    all_results = []
    labels = []
    start_time = time.time()
    for name, model in all_models:
        # start time for training
        # split the data based on the training split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training.split, random_state=training.random_state)
        
        # train the model with exception handling
        try:
            model.fit(X_train, y_train)
            yred =model.predict(X_test)
            # check all the metrics selected and save the results with exception handling
            results = {}
            if metrics.accuracy:
                results['train_accuracy'] = model.score(X_train, y_train) or -1
                results['test_accuracy'] =model.score(X_test, y_test) or -1
            if metrics.mse:
                results['mse'] = mean_squared_error(y_test, yred) or -1
            if metrics.rmse:
                results['rmse'] = np.sqrt(mean_squared_error(y_test, yred)) or -1
            if metrics.mae:
                results['mae'] = mean_absolute_error(y_test, yred) or  -1
        
            # check if the model is the best model
            if not best_results or best_results['test_accuracy'] < results['test_accuracy']:
                best_results = results
                best_model = model
            all_results.append(results)
                    # save the model to the database
            
            html_repr = estimator_html_repr(model)
            with open(f'assets/{name}.html', 'w', encoding='utf-8', errors='ignore') as f:
                f.write(html_repr)
            labels.append(name)
        except Exception as e:
            print(f'Error in training {name}: {e}')
            results = {'train_accuracy': -1, 'test_accuracy': -1, 'mse': -1, 'rmse': -1, 'mae': -1}
            all_results.append(results)
                    
        
    
    # load the best model and make visualization
    os.makedirs('media/models', exist_ok=True)
    dump(best_model, f'media/models/{name}.joblib')
    model = load(f'media/models/{name}.joblib')
    
    # bar chart for the results
    fig, ax = plt.subplots()
    trnacc = [result['train_accuracy'] for result in all_results]
    tstacc = [result['test_accuracy'] for result in all_results]
    mse = [result['mse'] for result in all_results]
    rmse = [result['rmse'] for result in all_results]
    mae = [result['mae'] for result in all_results]
    x = np.arange(len(trnacc))
    width = 0.35
    print(f'train accuracy: {trnacc}')
    print(f'test accuracy: {tstacc}')
    print(f'mse: {mse}')
    print(f'rmse: {rmse}')
    print(f'mae: {mae}')
    print(f'labels: {labels}')
    print(f'x: {x}')
    print('shapes', len(trnacc), len(tstacc), len(mse), len(rmse), len(mae), len(labels), len(x))
    print(all_results)
    ax.bar(x - width/2, trnacc, width, label='Train Accuracy', color='b', alpha=0.7)
    ax.bar(x + width/2, tstacc, width, label='Test Accuracy', color='r', alpha=0.7)

    ax.set_ylabel('Scores')
    ax.set_title('Scores by model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig('assets/metrics_acc.png')

    vis = Visualizations.objects.create(
        dataset=Dataset.objects.get(id=did),
        user=request.user,
        model=training,
        metric=metrics,
        algo=algos,
        plot='assets/metrics_acc.png'
    )
    
    # save the model to the database
    training.training_time = time.time() - start_time
    training.training_accuracy = best_results['train_accuracy']
    training.testing_accuracy = best_results['test_accuracy']
    training.model_path = f'media/models/{name}.joblib'
    
    training.save()
            
    return {
        'results': best_results,
        'best_model':f'/media/models/{name}.joblib',
        'graph': '/static/metrics_acc.png',
        'status': 'success',
    }

# classification pipeline
def classification_pipeline(request,df, did, pid, aid, mid, tid):
    pass


@csrf_exempt
def execute_pipeline(request):
    if request.method == 'POST':
        try:
            target = request.POST.get('target')
            did = request.session.get('current_dataset_id')
            pid = request.session.get('current_preprocess_id')
            aid = request.session.get('current_algorithm_id')
            mid = request.session.get('current_metric_id')
            tid = request.session.get('current_training_id')
            print(f'executing: {did}, {pid}, {aid}, {mid}, {tid}')
            dataset = Dataset.objects.get(id=did)
            df = pd.read_csv(dataset.file, encoding='latin-1', nrows=10000, skipinitialspace=True, skip_blank_lines=True, na_filter=True)
            print(df.head())
            print(df.columns)
            print(target)
            # check if the target is for classification or regression
            if df[target].dtype == 'object' or df[target].nunique() < 10 or df[target].dtype == 'bool':
                # classification
                if df[target].nunique() > 2:
                    results = classification_pipeline(request, df, did, pid, aid, mid, tid, target)
                else:
                    results = classification_pipeline(request, df, did, pid, aid, mid, tid, target)
            else:
                results = regression_pipeline(request, df, did, pid, aid, mid, tid, target)
            return JsonResponse(results)
        except Exception as e:
            print(e)
            return JsonResponse({'status': 'failed', 'error': f'⚠️ No Target parameter selected/ or error in training: {e}'})        
    else:
        return JsonResponse({'status': 'failed'})
    
    
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