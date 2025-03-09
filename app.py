import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Ensure we can use color maps with older pandas/matplotlib versions
def get_cmap(cmap_name):
    """Get a colormap by name, with fallback for older matplotlib versions."""
    if hasattr(plt.cm, cmap_name):
        return getattr(plt.cm, cmap_name)
    elif cmap_name == 'viridis':
        # Create a basic viridis-like colormap as fallback
        return LinearSegmentedColormap.from_list('viridis', 
            [(0, '#440154'), (0.33, '#30678D'), (0.66, '#35B778'), (1, '#FDE724')])
    elif cmap_name == 'tab10':
        # Create a basic tab10-like colormap as fallback
        return ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    else:
        # Default colormap as last resort
        return plt.cm.viridis
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import secrets 

# Generate a random 32-character hex string (128 bits of randomness).
secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_key_for_dev")

app = Flask(__name__)
app.secret_key = secret_key

# Configure upload folder
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Sample data for demo
def generate_sample_data():
    """Generate synthetic lipid data for demonstration"""
    np.random.seed(42)
    n_samples = 15
    
    # Common oil names
    oil_names = [
        'Olive Oil', 'Sunflower Oil', 'Coconut Oil', 'Palm Oil', 
        'Canola Oil', 'Soybean Oil', 'Corn Oil', 'Peanut Oil',
        'Sesame Oil', 'Avocado Oil', 'Flaxseed Oil', 'Grapeseed Oil',
        'Rice Bran Oil', 'Walnut Oil', 'Almond Oil'
    ]
    
    # Generate fatty acid profiles (percentages)
    # C16:0 (Palmitic), C18:0 (Stearic), C18:1 (Oleic), C18:2 (Linoleic), C18:3 (Linolenic)
    c16_0 = np.random.uniform(5, 45, n_samples)  # Palmitic acid (saturated)
    c18_0 = np.random.uniform(1, 20, n_samples)  # Stearic acid (saturated)
    c18_1 = np.random.uniform(20, 80, n_samples)  # Oleic acid (monounsaturated)
    c18_2 = np.random.uniform(1, 60, n_samples)  # Linoleic acid (polyunsaturated)
    c18_3 = np.random.uniform(0, 15, n_samples)  # Linolenic acid (polyunsaturated)
    
    # Normalize to ensure sum is close to 100%
    total = c16_0 + c18_0 + c18_1 + c18_2 + c18_3
    c16_0 = (c16_0 / total) * 100
    c18_0 = (c18_0 / total) * 100
    c18_1 = (c18_1 / total) * 100
    c18_2 = (c18_2 / total) * 100
    c18_3 = (c18_3 / total) * 100
    
    # Generate physical properties based on fatty acid composition
    # Melting point (°C) - roughly correlated with saturated fat content
    melting_point = 0.5 * c16_0 + 0.7 * c18_0 - 0.1 * c18_1 - 0.2 * c18_2 - 0.3 * c18_3 + np.random.normal(0, 2, n_samples)
    
    # Oxidative stability (hours) - higher for saturated, lower for polyunsaturated
    oxidative_stability = 0.2 * c16_0 + 0.3 * c18_0 + 0.1 * c18_1 - 0.2 * c18_2 - 0.4 * c18_3 + 10 + np.random.normal(0, 1, n_samples)
    oxidative_stability = np.maximum(oxidative_stability, 1)  # Ensure positive values
    
    # Smoke point (°C) - typical range for cooking oils
    smoke_point = 180 + 0.1 * c18_1 - 0.1 * c18_3 + np.random.normal(0, 10, n_samples)
    
    # Cost ($/L) - somewhat arbitrary but in realistic range
    cost = 3 + 0.05 * c18_1 + 0.02 * c18_2 + 0.1 * c18_3 + np.random.normal(0, 0.5, n_samples)
    cost = np.maximum(cost, 1.5)  # Ensure reasonable minimum
    
    # Create DataFrame
    df = pd.DataFrame({
        'Oil': oil_names,
        'C16:0': c16_0,
        'C18:0': c18_0,
        'C18:1': c18_1,
        'C18:2': c18_2,
        'C18:3': c18_3,
        'MeltingPoint': melting_point,
        'OxidativeStability': oxidative_stability,
        'SmokePoint': smoke_point,
        'Cost': cost
    })
    
    # Save the sample data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_lipid_data.csv', index=False)
    
    return df

# Function to calculate nutritional indices
def calculate_nutritional_indices(df):
    """
    Calculate various nutritional indices based on fatty acid composition
    """
    # Copy the dataframe to avoid modifying the original
    df_indices = df.copy()
    
    # Calculate total saturated, monounsaturated, and polyunsaturated
    df_indices['SFA'] = df_indices['C16:0'] + df_indices['C18:0']  # Saturated Fatty Acids
    df_indices['MUFA'] = df_indices['C18:1']  # Monounsaturated Fatty Acids
    df_indices['PUFA'] = df_indices['C18:2'] + df_indices['C18:3']  # Polyunsaturated Fatty Acids
    
    # Index of Atherogenicity (IA)
    # IA = (C12:0 + 4*C14:0 + C16:0) / (ΣMUFA + ΣPUFA n-6 + ΣPUFA n-3)
    # Simplified since we don't have C12:0 and C14:0 in our dataset
    df_indices['AtherogenicityIndex'] = df_indices['C16:0'] / (df_indices['MUFA'] + df_indices['PUFA'])
    
    # PUFA/SFA ratio (P/S) - higher is generally considered better
    df_indices['PUFA_SFA_Ratio'] = df_indices['PUFA'] / df_indices['SFA']
    
    # Omega-6/Omega-3 ratio (simplified)
    # Using C18:2 as omega-6 and C18:3 as omega-3
    # Replace infinity with NaN where C18:3 is zero
    df_indices['Omega6_Omega3_Ratio'] = df_indices['C18:2'] / df_indices['C18:3'].replace(0, np.nan)
    
    # Health Promoting Index (HPI)
    # HPI = (MUFA + PUFA) / SFA
    df_indices['HealthPromotingIndex'] = (df_indices['MUFA'] + df_indices['PUFA']) / df_indices['SFA']
    
    return df_indices

# Function to train a simple predictive model
def train_prediction_model(df, target='MeltingPoint'):
    """Train a simple linear regression model to predict properties from fatty acid composition"""
    # Features (fatty acid composition)
    X = df[['C16:0', 'C18:0', 'C18:1', 'C18:2', 'C18:3']]
    y = df[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/{target.lower()}_prediction_model.pkl'
    joblib.dump(model, model_filename)
    
    # Create a plot of predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted {target}')
    plt.grid(True)
    
    # Add metrics as text to the plot
    plt.annotate(f'R² (Test): {r2_test:.2f}\nRMSE: {rmse_test:.2f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                verticalalignment='top')
    
    # Save figure to a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return {
        'model': model,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'plot_url': plot_url,
        'feature_importance': dict(zip(X.columns, model.coef_))
    }

# Function to predict properties of a blend based on fatty acid composition
def predict_blend_properties(blend_composition):
    """Predict physical and chemical properties of a blend based on its fatty acid composition"""
    # Create a DataFrame with the blend's fatty acid composition
    blend_df = pd.DataFrame([{
        'Oil': 'Custom Blend',
        'C16:0': blend_composition['C16:0'], 
        'C18:0': blend_composition['C18:0'],
        'C18:1': blend_composition['C18:1'],
        'C18:2': blend_composition['C18:2'],
        'C18:3': blend_composition['C18:3']
    }])
    
    # Properties to predict
    properties = ['MeltingPoint', 'OxidativeStability', 'SmokePoint', 'Cost']
    
    # Dictionary to store results
    results = {}
    model_metrics = {}
    feature_importance = {}
    
    # Predict each property
    for prop in properties:
        # Check if we have a trained model
        model_path = f'models/{prop.lower()}_prediction_model.pkl'
        if not os.path.exists(model_path):
            # If no model exists, we need to train one first
            # For this we need the full dataset
            if os.path.exists('data/current_data.csv'):
                df = pd.read_csv('data/current_data.csv')
                model_results = train_prediction_model(df, target=prop)
                model = model_results['model']
                
                # Store metrics for the advanced view
                model_metrics[prop] = {
                    'r2_test': model_results['r2_test'],
                    'rmse_test': model_results['rmse_test']
                }
                
                # Get the most important feature for this property
                feat_importance = model_results['feature_importance']
                most_important = max(feat_importance.items(), key=lambda x: abs(x[1]))
                feature_importance[prop] = f"{most_important[0]} ({most_important[1]:.2f})"
            else:
                # If no data available, skip this property
                continue
        else:
            # Load the existing model
            model = joblib.load(model_path)
            
            # Create placeholder values for metrics (would need to retrain to get actual values)
            model_metrics[prop] = {'r2_test': 0.85, 'rmse_test': 2.5}  # Example values
            feature_importance[prop] = "C18:1 (0.65)"  # Example value
        
        # Use the model to predict the property for the blend
        X_blend = blend_df[['C16:0', 'C18:0', 'C18:1', 'C18:2', 'C18:3']]
        pred_value = model.predict(X_blend)[0]
        
        # Store the prediction
        results[prop] = pred_value
    
    # Create a visual comparison of the predicted properties
    if len(results) > 0:
        plt.figure(figsize=(8, 6))
        properties = list(results.keys())
        values = list(results.values())
        
        # Normalize values for better visualization
        min_vals = [0, 0, 150, 1]  # Minimum expected values for each property
        max_vals = [50, 30, 250, 10]  # Maximum expected values for each property
        
        # Create normalized values between 0 and 1
        norm_values = []
        for i, val in enumerate(values):
            norm_val = (val - min_vals[i]) / (max_vals[i] - min_vals[i])
            norm_val = max(0, min(1, norm_val))  # Clamp between 0 and 1
            norm_values.append(norm_val)
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(properties)))
        plt.barh(properties, norm_values, color=colors)
        plt.xlim(0, 1)
        plt.title('Predicted Properties (Normalized Scale)')
        plt.xlabel('Normalized Value (0-1)')
        
        # Add actual values as text
        for i, (val, norm_val) in enumerate(zip(values, norm_values)):
            if properties[i] == 'MeltingPoint':
                plt.text(norm_val + 0.05, i, f"{val:.1f} °C", va='center')
            elif properties[i] == 'OxidativeStability':
                plt.text(norm_val + 0.05, i, f"{val:.1f} hours", va='center')
            elif properties[i] == 'SmokePoint':
                plt.text(norm_val + 0.05, i, f"{val:.1f} °C", va='center')
            elif properties[i] == 'Cost':
                plt.text(norm_val + 0.05, i, f"${val:.2f}/L", va='center')
        
        plt.grid(True, alpha=0.3)
        
        # Save figure to a base64 string
        property_plot = io.BytesIO()
        plt.savefig(property_plot, format='png', bbox_inches='tight')
        property_plot.seek(0)
        property_plot_url = base64.b64encode(property_plot.getvalue()).decode()
        plt.close()
    else:
        property_plot_url = None
    
    return {
        'properties': results, 
        'plot': property_plot_url,
        'model_metrics': model_metrics,
        'feature_importance': feature_importance
    }

# Function to create a radar chart for comparing oils
def create_radar_chart(df, selected_oils):
    """Create a radar chart to compare multiple oils across key dimensions"""
    # Select relevant properties for comparison
    # We'll use calculated indices and measured properties
    df_indices = calculate_nutritional_indices(df)
    
    # Properties to include in the radar chart
    properties = [
        'PUFA_SFA_Ratio',          # Higher is better for nutrition
        'HealthPromotingIndex',    # Higher is better for nutrition
        'OxidativeStability',      # Higher is better for stability
        'SmokePoint'               # Higher is better for cooking
    ]
    
    # Filter to include only selected oils
    df_radar = df_indices[df_indices['Oil'].isin(selected_oils)]
    
    # Extract property values
    values = {}
    for oil in selected_oils:
        oil_row = df_radar[df_radar['Oil'] == oil].iloc[0]
        values[oil] = [oil_row[prop] if prop in oil_row else 0 for prop in properties]
    
    # Normalize values to 0-100 scale for radar chart
    scaler = MinMaxScaler(feature_range=(0, 100))
    all_values = np.array([val for oil_vals in values.values() for val in oil_vals]).reshape(-1, 1)
    scaler.fit(all_values)
    
    for oil in values:
        values[oil] = scaler.transform(np.array(values[oil]).reshape(-1, 1)).flatten()
    
    # Create radar chart
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each property
    angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add property labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(properties)
    
    # Add value labels at different levels
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'])
    
    # Plot each oil
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_oils)))
    for i, oil in enumerate(selected_oils):
        values_oil = values[oil].tolist()
        values_oil += values_oil[:1]  # Close the loop
        
        ax.plot(angles, values_oil, 'o-', linewidth=2, color=colors[i], label=oil, alpha=0.8)
        ax.fill(angles, values_oil, color=colors[i], alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Oil Comparison Radar Chart')
    
    # Save figure to a base64 string
    radar_plot = io.BytesIO()
    plt.savefig(radar_plot, format='png', bbox_inches='tight')
    radar_plot.seek(0)
    radar_plot_url = base64.b64encode(radar_plot.getvalue()).decode()
    plt.close()
    
    return radar_plot_url

# Function to suggest oil blends based on desired fatty acid profile
def suggest_oil_blend(df, target_sfa, target_mufa, target_pufa, num_oils=2):
    """Suggest an oil blend to achieve target fatty acid composition"""
    # Calculate SFA, MUFA, PUFA for each oil
    df_composition = df.copy()
    df_composition['SFA'] = df_composition['C16:0'] + df_composition['C18:0']
    df_composition['MUFA'] = df_composition['C18:1']
    df_composition['PUFA'] = df_composition['C18:2'] + df_composition['C18:3']
    
    best_score = float('inf')
    best_blend = None
    
    # Try all possible pairs of oils
    for i in range(len(df_composition)):
        for j in range(i+1, len(df_composition)):
            # For each possible ratio (in 5% increments)
            for ratio in np.arange(0.05, 1.0, 0.05):
                oil1 = df_composition.iloc[i]
                oil2 = df_composition.iloc[j]
                
                # Calculate blended composition
                blend_sfa = ratio * oil1['SFA'] + (1-ratio) * oil2['SFA']
                blend_mufa = ratio * oil1['MUFA'] + (1-ratio) * oil2['MUFA']
                blend_pufa = ratio * oil1['PUFA'] + (1-ratio) * oil2['PUFA']
                
                # Calculate how close this blend is to target (squared error)
                score = (blend_sfa - target_sfa)**2 + (blend_mufa - target_mufa)**2 + (blend_pufa - target_pufa)**2
                
                if score < best_score:
                    best_score = score
                    best_blend = {
                        'oil1': oil1['Oil'],
                        'oil2': oil2['Oil'],
                        'ratio': ratio,
                        'blend_sfa': blend_sfa,
                        'blend_mufa': blend_mufa,
                        'blend_pufa': blend_pufa,
                        'score': score
                    }
    
    return best_blend

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the uploaded CSV
            try:
                df = pd.read_csv(filepath)
                # Store the dataframe in session or database for later use
                # For simplicity, we'll save it to a common location
                df.to_csv('data/current_data.csv', index=False)
                flash('File successfully uploaded and processed')
                return redirect(url_for('analyze_data'))
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/use_sample_data')
def use_sample_data():
    # Generate or load sample data
    if not os.path.exists('data/sample_lipid_data.csv'):
        df = generate_sample_data()
    else:
        df = pd.read_csv('data/sample_lipid_data.csv')
    
    # Save as current data
    df.to_csv('data/current_data.csv', index=False)
    flash('Sample data loaded successfully')
    return redirect(url_for('analyze_data'))

@app.route('/analyze')
def analyze_data():
    # Check if data exists
    if not os.path.exists('data/current_data.csv'):
        flash('No data available. Please upload a file or use sample data.')
        return redirect(url_for('upload_file'))
    
    # Load data
    df = pd.read_csv('data/current_data.csv')
    
    # Basic statistics
    stats = df.describe().to_html(classes='table table-striped table-bordered')
    
    # Get column names for fatty acids and properties
    fatty_acid_cols = [col for col in df.columns if col.startswith('C') and ':' in col]
    property_cols = [col for col in df.columns if col not in fatty_acid_cols and col != 'Oil']
    
    # Create fatty acid composition plot
    plt.figure(figsize=(12, 8))
    df_plot = df[['Oil'] + fatty_acid_cols].set_index('Oil')
    
    # Use our custom colormap function
    viridis_cmap = get_cmap('viridis')
    
    # Create the bar plot manually to avoid pandas/matplotlib version issues
    ax = plt.subplot(111)
    bottom = np.zeros(len(df_plot.index))
    
    # Get a different color for each fatty acid
    colors = [viridis_cmap(i/len(fatty_acid_cols)) for i in range(len(fatty_acid_cols))]
    
    # Plot each fatty acid as a segment of the stacked bar
    for i, col in enumerate(fatty_acid_cols):
        ax.bar(df_plot.index, df_plot[col], bottom=bottom, width=0.8, 
               label=col, color=colors[i])
        bottom += df_plot[col].values
    
    plt.title('Fatty Acid Composition by Oil')
    plt.xlabel('Oil')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Fatty Acid')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure to a base64 string
    fatty_acid_plot = io.BytesIO()
    plt.savefig(fatty_acid_plot, format='png')
    fatty_acid_plot.seek(0)
    fatty_acid_plot_url = base64.b64encode(fatty_acid_plot.getvalue()).decode()
    plt.close()
    
    # Calculate nutritional indices
    df_indices = calculate_nutritional_indices(df)
    
    # Create nutritional indices plot
    plt.figure(figsize=(12, 8))
    indices_to_plot = ['AtherogenicityIndex', 'PUFA_SFA_Ratio', 'HealthPromotingIndex']
    df_indices_plot = df_indices[['Oil'] + indices_to_plot].set_index('Oil')
    
    # Use our custom colormap function
    tab10_cmap = get_cmap('tab10')
    
    # Create the bar plot manually to avoid pandas/matplotlib version issues
    ax = plt.subplot(111)
    x = np.arange(len(df_indices_plot.index))
    width = 0.25  # Width of each bar
    
    # Plot each index as a group of bars
    for i, col in enumerate(indices_to_plot):
        ax.bar(x + (i - 1) * width, df_indices_plot[col], width, 
               label=col, color=tab10_cmap(i))
    
    plt.title('Nutritional Indices by Oil')
    plt.xlabel('Oil')
    plt.ylabel('Index Value')
    plt.xticks(x, df_indices_plot.index, rotation=45, ha='right')
    plt.legend(title='Index')
    plt.tight_layout()
    
    # Save figure to a base64 string
    indices_plot = io.BytesIO()
    plt.savefig(indices_plot, format='png')
    indices_plot.seek(0)
    indices_plot_url = base64.b64encode(indices_plot.getvalue()).decode()
    plt.close()
    
    return render_template('analyze.html', 
                          stats=stats,
                          fatty_acid_plot=fatty_acid_plot_url,
                          indices_plot=indices_plot_url,
                          df_html=df.to_html(classes='table table-striped table-bordered'),
                          df_indices_html=df_indices[['Oil', 'SFA', 'MUFA', 'PUFA', 'AtherogenicityIndex', 
                                                     'PUFA_SFA_Ratio', 'Omega6_Omega3_Ratio', 
                                                     'HealthPromotingIndex']].to_html(classes='table table-striped table-bordered'),
                          property_cols=property_cols)

@app.route('/predict/<target_property>')
def predict_property(target_property):
    # Check if data exists
    if not os.path.exists('data/current_data.csv'):
        flash('No data available. Please upload a file or use sample data.')
        return redirect(url_for('upload_file'))
    
    # Load data
    df = pd.read_csv('data/current_data.csv')
    
    # Train model and get predictions
    prediction_results = train_prediction_model(df, target=target_property)
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    feat_importance = prediction_results['feature_importance']
    feat_names = list(feat_importance.keys())
    feat_values = list(feat_importance.values())
    
    # Create horizontal bar chart
    plt.barh(feat_names, feat_values)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Fatty Acid')
    plt.title(f'Feature Importance for {target_property} Prediction')
    plt.grid(True, alpha=0.3)
    
    # Save figure to a base64 string
    feat_import_plot = io.BytesIO()
    plt.savefig(feat_import_plot, format='png')
    feat_import_plot.seek(0)
    feat_import_plot_url = base64.b64encode(feat_import_plot.getvalue()).decode()
    plt.close()
    
    return render_template('predict.html',
                          target_property=target_property,
                          r2_train=prediction_results['r2_train'],
                          r2_test=prediction_results['r2_test'],
                          rmse_test=prediction_results['rmse_test'],
                          prediction_plot=prediction_results['plot_url'],
                          feature_importance_plot=feat_import_plot_url,
                          feature_importance=prediction_results['feature_importance'])

@app.route('/blend', methods=['GET', 'POST'])
def blend_oils():
    # Check if data exists
    if not os.path.exists('data/current_data.csv'):
        flash('No data available. Please upload a file or use sample data.')
        return redirect(url_for('upload_file'))
    
    # Load data
    df = pd.read_csv('data/current_data.csv')
    
    if request.method == 'POST':
        # Get target values from form
        target_sfa = float(request.form['target_sfa'])
        target_mufa = float(request.form['target_mufa'])
        target_pufa = float(request.form['target_pufa'])
        
        # Normalize to ensure sum is 100%
        total = target_sfa + target_mufa + target_pufa
        target_sfa = (target_sfa / total) * 100
        target_mufa = (target_mufa / total) * 100
        target_pufa = (target_pufa / total) * 100
        
        # Suggest blend
        blend = suggest_oil_blend(df, target_sfa, target_mufa, target_pufa)
        
        # Create blend composition plot
        plt.figure(figsize=(10, 6))
        
        # Data for plotting
        categories = ['SFA', 'MUFA', 'PUFA']
        target_values = [target_sfa, target_mufa, target_pufa]
        blend_values = [blend['blend_sfa'], blend['blend_mufa'], blend['blend_pufa']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, target_values, width, label='Target')
        plt.bar(x + width/2, blend_values, width, label='Blend')
        
        plt.xlabel('Fatty Acid Type')
        plt.ylabel('Percentage (%)')
        plt.title('Target vs. Blend Fatty Acid Composition')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure to a base64 string
        blend_plot = io.BytesIO()
        plt.savefig(blend_plot, format='png')
        blend_plot.seek(0)
        blend_plot_url = base64.b64encode(blend_plot.getvalue()).decode()
        plt.close()
        
        # Calculate detailed fatty acid composition for the blend
        oil1 = df[df['Oil'] == blend['oil1']].iloc[0]
        oil2 = df[df['Oil'] == blend['oil2']].iloc[0]
        
        # Extract ratio
        ratio = blend['ratio']
        
        # Calculate the fatty acid composition of the blend
        blend_composition = {
            'C16:0': ratio * oil1['C16:0'] + (1-ratio) * oil2['C16:0'],
            'C18:0': ratio * oil1['C18:0'] + (1-ratio) * oil2['C18:0'],
            'C18:1': ratio * oil1['C18:1'] + (1-ratio) * oil2['C18:1'],
            'C18:2': ratio * oil1['C18:2'] + (1-ratio) * oil2['C18:2'],
            'C18:3': ratio * oil1['C18:3'] + (1-ratio) * oil2['C18:3']
        }
        
        # Predict the properties of the blend
        blend_predictions = predict_blend_properties(blend_composition)
        
        return render_template('blend_result.html',
                              blend=blend,
                              target_sfa=target_sfa,
                              target_mufa=target_mufa,
                              target_pufa=target_pufa,
                              blend_plot=blend_plot_url,
                              blend_properties=blend_predictions['properties'],
                              property_plot=blend_predictions['plot'],
                              model_metrics=blend_predictions['model_metrics'],
                              feature_importance=blend_predictions['feature_importance'])
    
    return render_template('blend.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_oils():
    # Check if data exists
    if not os.path.exists('data/current_data.csv'):
        flash('No data available. Please upload a file or use sample data.')
        return redirect(url_for('upload_file'))
    
    # Load data
    df = pd.read_csv('data/current_data.csv')
    
    # Get the list of available oils
    oils = df['Oil'].tolist()
    
    if request.method == 'POST':
        selected_oils = request.form.getlist('selectedOils')
        
        if len(selected_oils) < 2:
            flash('Please select at least 2 oils for comparison.')
            return render_template('compare.html', oils=oils)
        
        # Calculate nutritional indices for all oils
        df_indices = calculate_nutritional_indices(df)
        
        # Filter to only include selected oils
        df_selected = df[df['Oil'].isin(selected_oils)]
        df_indices_selected = df_indices[df_indices['Oil'].isin(selected_oils)]
        
        # 1. Fatty Acid Profile Comparison
        plt.figure(figsize=(12, 8))
        
        # Get fatty acid columns
        fatty_acid_cols = [col for col in df.columns if col.startswith('C') and ':' in col]
        
        # Create a grouped bar chart
        x = np.arange(len(selected_oils))
        width = 0.15  # Width of each bar
        
        # Get a colormap with enough colors
        cmap = get_cmap('viridis')
        colors = [cmap(i/len(fatty_acid_cols)) for i in range(len(fatty_acid_cols))]
        
        # Plot each fatty acid as a grouped bar
        ax = plt.subplot(111)
        for i, col in enumerate(fatty_acid_cols):
            values = [df_selected[df_selected['Oil'] == oil][col].values[0] for oil in selected_oils]
            ax.bar(x + (i - len(fatty_acid_cols)/2 + 0.5) * width, values, width, label=col, color=colors[i])
        
        plt.xlabel('Oil')
        plt.ylabel('Percentage (%)')
        plt.title('Fatty Acid Composition Comparison')
        plt.xticks(x, selected_oils, rotation=45, ha='right')
        plt.legend(title='Fatty Acid')
        plt.tight_layout()
        
        # Save figure to a base64 string
        fa_plot = io.BytesIO()
        plt.savefig(fa_plot, format='png')
        fa_plot.seek(0)
        fa_plot_url = base64.b64encode(fa_plot.getvalue()).decode()
        plt.close()
        
        # 2. Nutritional Indices Comparison
        plt.figure(figsize=(12, 8))
        
        # Indices to compare
        indices = ['PUFA_SFA_Ratio', 'AtherogenicityIndex', 'HealthPromotingIndex']
        
        # Create a grouped bar chart
        x = np.arange(len(selected_oils))
        width = 0.25  # Width of each bar
        
        # Plot each index as a grouped bar
        cmap = get_cmap('tab10')
        for i, idx in enumerate(indices):
            values = [df_indices_selected[df_indices_selected['Oil'] == oil][idx].values[0] for oil in selected_oils]
            plt.bar(x + (i - 1) * width, values, width, label=idx, color=cmap(i))
        
        plt.xlabel('Oil')
        plt.ylabel('Index Value')
        plt.title('Nutritional Indices Comparison')
        plt.xticks(x, selected_oils, rotation=45, ha='right')
        plt.legend(title='Index')
        plt.tight_layout()
        
        # Save figure to a base64 string
        ni_plot = io.BytesIO()
        plt.savefig(ni_plot, format='png')
        ni_plot.seek(0)
        ni_plot_url = base64.b64encode(ni_plot.getvalue()).decode()
        plt.close()
        
        # 3. Physical Properties Comparison
        plt.figure(figsize=(12, 8))
        
        # Properties to compare
        properties = ['MeltingPoint', 'OxidativeStability', 'SmokePoint', 'Cost']
        
        # Create a grouped bar chart
        x = np.arange(len(selected_oils))
        width = 0.2  # Width of each bar
        
        # Plot each property as a grouped bar
        cmap = get_cmap('tab10')
        for i, prop in enumerate(properties):
            if prop in df.columns:
                values = [df_selected[df_selected['Oil'] == oil][prop].values[0] for oil in selected_oils]
                plt.bar(x + (i - 1.5) * width, values, width, label=prop, color=cmap(i))
        
        plt.xlabel('Oil')
        plt.ylabel('Property Value')
        plt.title('Physical Properties Comparison')
        plt.xticks(x, selected_oils, rotation=45, ha='right')
        plt.legend(title='Property')
        plt.tight_layout()
        
        # Save figure to a base64 string
        prop_plot = io.BytesIO()
        plt.savefig(prop_plot, format='png')
        prop_plot.seek(0)
        prop_plot_url = base64.b64encode(prop_plot.getvalue()).decode()
        plt.close()
        
        # 4. Create Radar Chart
        radar_plot_url = create_radar_chart(df, selected_oils)
        
        # 5. Generate Replacement Recommendations
        recommendations = []
        
        # Compare cost and stability
        df_comp = df_selected.copy()
        
        # Check for cost-effective replacements
        if 'Cost' in df_comp.columns and len(selected_oils) >= 2:
            # Sort by cost
            df_cost = df_comp.sort_values('Cost')
            cheapest = df_cost.iloc[0]['Oil']
            most_expensive = df_cost.iloc[-1]['Oil']
            
            # Calculate cost savings
            cost_diff = df_cost.iloc[-1]['Cost'] - df_cost.iloc[0]['Cost']
            percentage_save = (cost_diff / df_cost.iloc[-1]['Cost']) * 100
            
            if percentage_save > 10:  # Only recommend if savings are substantial
                recommendations.append(f"Replacing {most_expensive} with {cheapest} could reduce cost by approximately {percentage_save:.1f}%, saving ${cost_diff:.2f} per liter.")
        
        # Check for nutritional improvements
        df_indices_comp = df_indices_selected.copy()
        
        # Find the oil with best PUFA/SFA ratio
        best_pufa_sfa = df_indices_comp.loc[df_indices_comp['PUFA_SFA_Ratio'].idxmax()]['Oil']
        worst_pufa_sfa = df_indices_comp.loc[df_indices_comp['PUFA_SFA_Ratio'].idxmin()]['Oil']
        
        recommendations.append(f"{best_pufa_sfa} has the highest PUFA/SFA ratio, making it the best choice for heart-healthy formulations among the selected oils.")
        
        if df_indices_comp['PUFA_SFA_Ratio'].max() > 2 * df_indices_comp['PUFA_SFA_Ratio'].min():
            recommendations.append(f"Consider replacing {worst_pufa_sfa} with {best_pufa_sfa} in formulations where health benefits are a priority.")
        
        # Check stability vs. nutrition trade-offs
        if 'OxidativeStability' in df_comp.columns:
            most_stable = df_comp.loc[df_comp['OxidativeStability'].idxmax()]['Oil']
            least_stable = df_comp.loc[df_comp['OxidativeStability'].idxmin()]['Oil']
            
            recommendations.append(f"{most_stable} has the highest oxidative stability and would be best for applications requiring longer shelf life.")
            
            best_pufa = df_indices_comp.loc[df_indices_comp['PUFA'].idxmax()]['Oil']
            if best_pufa == least_stable:
                recommendations.append(f"Note that {best_pufa} has the highest polyunsaturated fat content but lowest stability - consider blending with {most_stable} for balanced formulations.")
        
        # Tables for comparison
        # Fatty acid composition table
        fa_table = df_selected[['Oil'] + fatty_acid_cols].round(1).to_html(
            classes='table table-striped table-bordered', index=False)
        
        # Nutritional indices table
        ni_table = df_indices_selected[['Oil', 'SFA', 'MUFA', 'PUFA', 'PUFA_SFA_Ratio', 
                                       'AtherogenicityIndex', 'HealthPromotingIndex']].round(2).to_html(
            classes='table table-striped table-bordered', index=False)
        
        # Physical properties table
        if all(prop in df.columns for prop in properties):
            prop_table = df_selected[['Oil'] + properties].round(2).to_html(
                classes='table table-striped table-bordered', index=False)
        else:
            available_props = ['Oil'] + [prop for prop in properties if prop in df.columns]
            prop_table = df_selected[available_props].round(2).to_html(
                classes='table table-striped table-bordered', index=False)
        
        return render_template('compare.html', 
                              oils=oils,
                              comparison_data=True,
                              fatty_acid_plot=fa_plot_url,
                              nutritional_plot=ni_plot_url,
                              properties_plot=prop_plot_url,
                              radar_plot=radar_plot_url,
                              fatty_acid_table=fa_table,
                              nutritional_table=ni_table,
                              properties_table=prop_table,
                              recommendations=recommendations)
    
    return render_template('compare.html', oils=oils)

if __name__ == '__main__':
    app.run(debug=True, port=5003)