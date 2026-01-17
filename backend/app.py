import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
import io
import base64
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error, classification_report
)

app = Flask(__name__)
CORS(app)

# Define absolute path for model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.joblib')

model = None
ct = None
label_encoder = None
problem_type = None
training_columns = None
training_dtypes = None

# Helper Functions

def safe_feature_importances(model):
    """Safely retrieve feature importances from a model, handling exceptions."""
    try:
        fi = model.feature_importances_
        return [round(x, 5) for x in fi.tolist()]
    except Exception:
        return "Not available"

def plot_confusion_matrix(cm):
    """Return a base64-encoded PNG of the confusion matrix."""
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_decision_tree(model, n_rows):
    """Return a base64-encoded PNG of the decision tree with scroll-friendly dimensions."""
    depth = model.get_depth() if hasattr(model, 'get_depth') else 10
    n_nodes = model.tree_.node_count if hasattr(model, 'tree_') else 50
    
    fig_width = min(40, max(20, n_nodes // 5))
    fig_height = min(60, max(15, depth * 2))
    
    plt.figure(figsize=(fig_width, fig_height), dpi=150, facecolor='white')
    
    plot_tree(model,
             filled=True,
             rounded=True,
             proportion=True,
             fontsize=14,
             node_ids=False,
             impurity=False,
             class_names=True,
             precision=2,
             feature_names=model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None,
             label='all',
             max_depth=8 if depth > 8 else None,
             ax=plt.gca())

    plt.title("Decision Tree Visualization", fontsize=16, pad=20)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_scatter_regression(y_test, y_pred):
    """Return a base64-encoded PNG scatter plot comparing actual vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Scatter Plot: Actual vs Predicted")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_roc_curve(model, X_test, y_test):
    """Return a base64-encoded PNG of the ROC curve for logistic regression."""
    from sklearn.metrics import roc_curve, auc
    # Check if binary classification
    if len(np.unique(y_test)) == 2:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Logistic Regression")
        plt.legend(loc="lower right")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return {
            'image': base64.b64encode(buf.read()).decode('utf-8')
        }
    else:
        return None  # ROC curve not plotted for multiclass

def plot_decision_boundary(model, X, y):
    """Return a base64-encoded PNG of the decision boundary for SVC (requires 2 features)."""
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plot requires exactly 2 features.")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVC Decision Boundary")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_svr_fit(X, y, model):
    """Return a base64-encoded PNG of the fitted curve for SVR regression when there's one feature."""
    X_flat = X.flatten()
    x_range = np.linspace(X_flat.min(), X_flat.max(), 500).reshape(-1, 1)
    y_pred_line = model.predict(x_range)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_flat, y, color='blue', alpha=0.5, label='Data')
    plt.plot(x_range, y_pred_line, color='red', linewidth=2, label='SVR Fit')
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("SVR Fitted Curve")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_svr_surface(X, y, model):
    """Return a base64-encoded PNG of the regression surface for SVR when there are 2 features."""
    if X.shape[1] != 2:
        raise ValueError("SVR surface plot requires exactly 2 features.")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(xx, yy, Z, alpha=0.7, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVR Regression Surface")
    plt.colorbar(cp)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_feature_importance(model, feature_names=None, top_n=20):
    """Return a base64-encoded PNG of feature importances for ensemble models."""
    try:
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {type(model).__name__} does not have feature_importances_ attribute")
            return None
            
        importances = model.feature_importances_
        
        if importances is None or len(importances) == 0:
            print("Feature importances are empty")
            return None
            
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Get top N features
        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(indices)), importances[indices], color='steelblue')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return {
            'image': base64.b64encode(buf.read()).decode('utf-8')
        }
    except Exception as e:
        print(f"Error in plot_feature_importance: {str(e)}")
        return None

def plot_residuals(y_test, y_pred):
    """Return a base64-encoded PNG of residual plot for regression."""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Residual scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    
    # Residual histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return {
        'image': base64.b64encode(buf.read()).decode('utf-8')
    }

def plot_precision_recall_curve(model, X_test, y_test):
    """Return a base64-encoded PNG of the Precision-Recall curve."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Check if binary classification
    if len(np.unique(y_test)) == 2:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        avg_precision = average_precision_score(y_test, y_score)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return {
            'image': base64.b64encode(buf.read()).decode('utf-8')
        }
    return None

# Routes
@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze the uploaded file to determine problem type, features, and suitable models."""
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        if df.shape[0] < 1:
            return jsonify({'error': 'File is empty!'}), 400

        # Unified data handling for all datasets
        last_col = df.iloc[:, -1]
        
        # Determine problem type based on target column dtype
        if pd.api.types.is_object_dtype(last_col) or pd.api.types.is_integer_dtype(last_col):
            # Check if it's classification (categorical or integers with limited unique values)
            unique_ratio = len(last_col.unique()) / len(last_col)
            if unique_ratio < 0.05 or pd.api.types.is_object_dtype(last_col):
                problem = 'classification'
                models = [
                    'logistic_regression', 'knn', 'naive_bayes',
                    'decision_tree', 'svm'
                ]
            else:
                problem = 'regression'
                models = [
                    'linear_regression', 'ridge', 'lasso', 'elasticnet',
                    'decision_tree', 'svm'
                ]
        else:
            problem = 'regression'
            models = [
                'linear_regression', 'ridge', 'lasso', 'elasticnet',
                'decision_tree', 'svm'
            ]
        
        features = list(df.columns)
        dataset_type = 'tabular'
        
        return jsonify({
            'problem_type': problem,
            'columns': features,
            'sample_count': df.shape[0],
            'models': models,
            'dataset_type': dataset_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train a machine learning model based on the uploaded dataset and selected algorithm."""
    global model, ct, label_encoder, problem_type, training_columns, training_dtypes
    try:
        file = request.files['file']
        algorithm = request.form.get('algorithm')
        show_metrics = request.form.get('show_metrics', 'false').lower() == 'true'
        
        df = pd.read_csv(file)
        if df.shape[0] < 2:
            return jsonify({'error': 'Dataset must have at least two rows!'}), 400

        # Unified Dataset Handling (works for all datasets including MNIST)
        query_row = df.iloc[[-1]].copy()
        train_df = df.iloc[:-1].copy()
        train_df.dropna(inplace=True)
        if train_df.empty:
            return jsonify({'error': 'Training data is empty after dropping null values!'}), 400
        
        # Sample large datasets for performance (increased limit to 15000)
        if train_df.shape[0] > 15000:
            train_df = train_df.sample(n=15000, random_state=42)
        
        X = train_df.iloc[:, :-1]
        y = train_df.iloc[:, -1]
        training_columns = list(X.columns)
        training_dtypes = X.dtypes.to_dict()
        
        # Determine problem type and encode target variable
        if pd.api.types.is_object_dtype(y):
            problem_type = 'classification'
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        elif pd.api.types.is_integer_dtype(y):
            # Check if integer column is categorical (classification) or continuous (regression)
            unique_ratio = len(y.unique()) / len(y)
            if unique_ratio < 0.05:  # Less than 5% unique values = classification
                problem_type = 'classification'
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                problem_type = 'regression'
                y_encoded = y.values
        else:
            problem_type = 'regression'
            y_encoded = y.values
        
        # Handle categorical features in X
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            ct = ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
                remainder='passthrough'
            )
            X_transformed = ct.fit_transform(X)
        else:
            X_transformed = X.values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y_encoded, test_size=0.2, random_state=42
        )
        
        # Prepare sample for query prediction
        sample_df = query_row.iloc[:, :-1].reindex(columns=training_columns, fill_value=np.nan)
        sample = ct.transform(sample_df) if ct is not None else sample_df.values
        n_rows_for_plot = train_df.shape[0]
        
        # Define Algorithms
        if problem_type == 'classification':
            supervised_algorithms = {
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
                'svm': SVC(probability=True)
            }
        else:
            supervised_algorithms = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=1.0),
                'elasticnet': ElasticNet(alpha=1.0, l1_ratio=0.5),
                'decision_tree': DecisionTreeRegressor(max_depth=5, random_state=42),
                'svm': SVR()
            }
        
        model = supervised_algorithms.get(algorithm)
        if not model:
            return jsonify({'error': f'Invalid algorithm for {problem_type}!'}), 400
        
        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute Metrics
        if problem_type == 'classification':
            acc = round(accuracy_score(y_test, y_pred) * 100, 5)
            prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 5)
            rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 5)
            f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 5)
            cm = confusion_matrix(y_test, y_pred)
            cls_report = classification_report(y_test, y_pred, zero_division=0)
            
            metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'classification_report': cls_report
            }
            if show_metrics:
                if algorithm == 'logistic_regression':
                    plot_data = plot_roc_curve(model, X_test, y_test)
                    if plot_data:
                        metrics['plot'] = plot_data['image']
                    else:
                        plot_data = plot_confusion_matrix(cm)
                        metrics['plot'] = plot_data['image']
                elif algorithm == 'svm':
                    if X_test.shape[1] == 2:
                        plot_data = plot_decision_boundary(model, X_test, y_test)
                        metrics['plot'] = plot_data['image']
                    else:
                        plot_data = plot_confusion_matrix(cm)
                        metrics['plot'] = plot_data['image']
                elif algorithm == 'decision_tree':
                    plot_data = plot_decision_tree(model, n_rows_for_plot)
                    metrics['plot'] = plot_data['image']
                else:
                    plot_data = plot_confusion_matrix(cm)
                    metrics['plot'] = plot_data['image']
        else:
            mse = round(mean_squared_error(y_test, y_pred), 5)
            r2 = round(r2_score(y_test, y_pred), 5)
            mae = round(mean_absolute_error(y_test, y_pred), 5)
            
            metrics = {'mse': mse}
            if show_metrics:
                metrics.update({
                    'r2_score': r2,
                    'mean_absolute_error': mae
                })
            # Linear models: show coefficients and residual plots
            if algorithm in ['linear_regression', 'ridge', 'lasso', 'elasticnet']:
                try:
                    coeffs = model.coef_.tolist()
                    coefficients = ", ".join(str(round(x, 5)) for x in coeffs)
                    metrics['coefficients'] = coefficients
                except Exception as e:
                    metrics['coefficients'] = f"Not available: {str(e)}"
                # Show residual plot for linear models
                plot_data = plot_residuals(y_test, y_pred)
                metrics['plot'] = plot_data['image']
            elif algorithm == 'svm':
                if X_test.shape[1] == 1:
                    plot_data = plot_svr_fit(X_test, y_test, model)
                    metrics['plot'] = plot_data['image']
                elif X_test.shape[1] == 2:
                    plot_data = plot_svr_surface(X_test, y_test, model)
                    metrics['plot'] = plot_data['image']
                else:
                    plot_data = plot_scatter_regression(y_test, y_pred)
                    metrics['plot'] = plot_data['image']
            elif algorithm == 'decision_tree':
                if show_metrics:
                    plot_data = plot_decision_tree(model, n_rows_for_plot)
                    metrics['plot'] = plot_data['image']
        
        # Generate Prediction for Query Sample (already prepared above)
        
        prediction = model.predict(sample)
        if problem_type == 'classification' and label_encoder is not None:
            prediction = label_encoder.inverse_transform(prediction)
        
        # Save Model
        joblib.dump({
            'model': model,
            'ct': ct,
            'label_encoder': label_encoder,
            'problem_type': problem_type,
            'training_columns': training_columns,
            'training_dtypes': training_dtypes
        }, MODEL_PATH)
        
        return jsonify({
            'message': 'Model trained successfully!',
            'metrics': metrics,
            'query_prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model."""
    global training_dtypes
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Model not found. Please train a model first.'}), 400
        
        data = request.json
        features = data['features']
        
        loaded_data = joblib.load(MODEL_PATH)
        model = loaded_data['model']
        ct = loaded_data['ct']
        label_encoder = loaded_data['label_encoder']
        problem_type = loaded_data['problem_type']
        training_columns = loaded_data.get('training_columns')
        training_dtypes = loaded_data.get('training_dtypes')
        
        if training_columns:
            # Create a DataFrame with the input features
            features_df = pd.DataFrame([features], columns=training_columns)
            
            # Convert feature types to match training data
            for col in training_columns:
                if col not in features_df.columns:
                    features_df[col] = np.nan
                else:
                    dtype = training_dtypes.get(col)
                    if pd.api.types.is_numeric_dtype(dtype):
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                        features_df[col] = features_df[col].astype('category')
                    else:
                        features_df[col] = features_df[col].astype(dtype)
            
            # Check for missing or invalid values
            if features_df.isnull().values.any():
                return jsonify({'error': 'Invalid or missing values in input features.'}), 400
            
            features_processed = ct.transform(features_df) if ct is not None else features_df.values
        else:
            features_processed = np.array([features])
        
        prediction = model.predict(features_processed)
        if problem_type == 'classification' and label_encoder is not None:
            prediction = label_encoder.inverse_transform(prediction)
        
        return jsonify({'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else round(prediction,5)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)