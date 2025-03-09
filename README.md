# lipids_demo
##  Proof of Concept Demo

A proof-of-concept web platform that integrates lipid science with data analytics to help researchers and product developers understand how lipid structure affects chemical, physical, and nutritional properties.

## Overview

This platform serves as a prototype for an interdisciplinary PhD project that combines lipid science with data science to develop AI tools that can identify relationships between lipid structure, properties, and performance in food products.

![Screenshot of the platform](static/screenshots/platform-demo.png)

## Features

- **Data Upload & Management**: Upload your own lipid data or use the built-in sample dataset
- **Nutritional Analysis**: Calculate indices like atherogenicity index, PUFA/SFA ratio, and more
- **Fatty Acid Visualization**: Visualize the fatty acid composition of different oils
- **Property Prediction**: Predict physical properties (melting point, stability) from fatty acid composition
- **Oil Blending Tool**: Formulate oil blends with specific fatty acid profiles

## Setup & Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Git

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/JulianMason/lipids
cd lipid-ai-platform
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to:
```
http://localhost:5003
```

## How to Use

### 1. Data Upload
- Start by uploading your lipid data in CSV format or use the sample dataset
- Required columns include oil names and fatty acid compositions (C16:0, C18:0, etc.)
- Optional: include measured properties for prediction validation

### 2. Analysis Dashboard
- View fatty acid composition visualizations
- Review nutritional indices (atherogenicity, PUFA/SFA ratios)
- Explore descriptive statistics for your dataset

### 3. Property Prediction
- Select a property to predict (melting point, oxidative stability, etc.)
- View prediction model performance metrics
- Understand which fatty acids have the greatest impact on each property

### 4. Oil Blending
- Input your desired fatty acid profile (SFA, MUFA, PUFA percentages)
- Get recommendations for oil blends that match your target profile
- Review applications for your custom blend

## Project Structure

```
lipid-ai-platform/
├── app.py                # Main Flask application
├── data/                 # Data storage directory
│   └── sample_lipid_data.csv  # Sample dataset
├── models/               # Saved prediction models
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Future Potential Integrations

This prototype demonstrates the core concept, but a full implementation could include:

- **Advanced Modeling**: Non-linear models, deep learning for property prediction
- **Lipidomics Integration**: Connect to external lipidomics databases
- **Experimental Data**: Incorporate lab results for model validation
- **Cost Optimization**: Economic analysis for oil blend formulation
- **Sensory Properties**: Predict taste, aroma, and texture characteristics
- **Health Impact**: Deeper analysis of nutritional and health effects

## Contributing

This is a research prototype, but contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
