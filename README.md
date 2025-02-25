
# Train_easy

![Train Easy Screenshot](/screenshots/1.png)

## Overview

Train_easy is a Django-based web application designed to simplify the process of training machine learning models. It provides a user-friendly interface to upload datasets, select preprocessing steps, choose algorithms, and evaluate model performance with various metrics.

## Features

- **Dataset Management**: Upload and manage datasets.
- **Preprocessing**: Select preprocessing steps like normalization, encoding, imputation, feature selection, and PCA.
- **Algorithm Selection**: Choose from various algorithms including Linear Models, Decision Trees, Random Forests, SVM, Naive Bayes, and KNN.
- **Metric Evaluation**: Evaluate models using metrics like accuracy, MSE, RMSE, MAE, confusion matrix, ROC AUC, precision, recall, and F1 score.
- **Model Training**: Train models with selected configurations and visualize the results.
- **Model Management**: View, download, and delete trained models.

## Tech Stack

- **Backend**: Django 5.0.3 (Python 3.x)
- **Frontend**: HTML, CSS, JavaScript
- **Third-party Libraries**: `crispy_forms`, `crispy_bootstrap5`
- **Machine Learning Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zaid-kamil/Train_easy.git
   cd Train_easy
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

5. **Run the development server**
   ```bash
   python manage.py runserver
   ```

## Usage

1. **Upload Dataset**: Navigate to the dataset upload page and upload your dataset.
2. **Select Preprocessing**: Choose the preprocessing steps you want to apply.
3. **Choose Algorithm**: Select the algorithms you want to use for training.
4. **Select Metrics**: Choose the metrics to evaluate the model performance.
5. **Train Model**: Start the training process and visualize the results.
6. **Manage Models**: View, download, or delete trained models from the management interface.

## Screenshots
<img src="/screenshots/1.png" alt="Screenshot 1" style="width:45%; display:inline-block; margin-right:4%;" />
<img src="/screenshots/1.png" alt="Screenshot 2" style="width:45%; display:inline-block; margin-right:4%;" />

<img src="/screenshots/1.png" alt="Screenshot 3" style="width:45%; display:inline-block; margin-right:4%;" />
<img src="/screenshots/1.png" alt="Screenshot 4" style="width:45%; display:inline-block; margin-right:4%;" />

<img src="/screenshots/1.png" alt="Screenshot 5" style="width:45%; display:inline-block; margin-right:4%;" />
<img src="/screenshots/1.png" alt="Screenshot 6" style="width:45%; display:inline-block; margin-right:4%;" />


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or support, please contact [zaid-kamil](https://github.com/zaid-kamil).
