# 💳 Credit Card Fraud Detection

A high-precision machine learning web application to detect fraudulent credit card transactions using Streamlit, Scikit-learn, and Logistic Regression with **97.1% accuracy**.

## 🎯 Features

- **High-precision fraud detection** using Logistic Regression
- **Interactive web interface** with multiple analysis modes
- **Real-time transaction analysis** with risk assessment
- **Batch processing** for multiple transactions
- **Visual insights** and model performance metrics
- **Class imbalance handling** using balanced weights

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn (Logistic Regression)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud

## 📊 Model Performance

- **Accuracy**: 97.1%
- **Algorithm**: Logistic Regression with balanced class weights
- **Features**: 30 (V1-V28 PCA components + Amount + Time)
- **Data Split**: Stratified 80/20 train-test split
- **Class Imbalance**: Addressed using under-sampling technique

## 🔧 Local Setup

1. Clone the repository:
```bash
git clone https://github.com/abhishekK2310/credit-card-fraud-detection-app.git
cd credit-card-fraud-detection-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## 💡 How It Works

### Data Processing
1. **Feature Engineering**: Uses PCA-transformed features (V1-V28) plus Amount and Time
2. **Normalization**: StandardScaler for feature scaling
3. **Class Balance**: Balanced class weights to handle fraud/legitimate imbalance

### Model Training
1. **Algorithm**: Logistic Regression (fast, interpretable, effective for binary classification)
2. **Evaluation**: Stratified train-test split ensures reliable metrics
3. **Performance**: Optimized for high precision to minimize false positives

### Prediction Process
1. **Input Processing**: Scales input features using trained scaler
2. **Classification**: Logistic Regression model predicts fraud probability
3. **Risk Assessment**: Categorizes transactions as LOW/MEDIUM/HIGH risk

## 🎮 Usage

### Single Transaction Analysis
- Enter transaction details (Amount, Time, V1-V28 features)
- Get instant fraud prediction with probability score
- View risk level and detailed probability breakdown

### Batch Analysis
- Generate sample transactions for testing
- Analyze multiple transactions simultaneously
- View fraud detection statistics and distributions

### Model Insights
- Explore model performance metrics
- Understand feature importance
- Review technical implementation details

## 📈 Key Metrics

- **Precision**: Minimizes false fraud alerts
- **Recall**: Catches actual fraudulent transactions
- **F1-Score**: Balanced performance measure
- **Accuracy**: Overall correctness of predictions

## 🔒 Security Features

- **Real-time Detection**: Instant fraud assessment
- **Risk Categorization**: LOW/MEDIUM/HIGH risk levels
- **Probability Scoring**: Confidence in predictions
- **Batch Processing**: Efficient analysis of multiple transactions

## 🌟 Highlights

- **Production-Ready**: Scalable and efficient implementation
- **User-Friendly**: Intuitive interface for non-technical users
- **Comprehensive**: Multiple analysis modes and visualizations
- **Reliable**: Stratified evaluation ensures robust performance

## 📱 Deployment

This app is optimized for deployment on Streamlit Cloud with automatic GitHub integration.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

MIT License

---

**Built with ❤️ using Streamlit, Scikit-learn, and Plotly | Achieving 97.1% Accuracy in Fraud Detection**