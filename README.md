# 📈 Stock Price Forecasting using ARIMA

A lightweight and interactive **Streamlit app** for forecasting stock prices using the **ARIMA (AutoRegressive Integrated Moving Average)** model. Visualize trends, check stationarity, and predict future prices—all within a user-friendly UI.

--------------------------

![App Demo](https://github.com/BBhuwanJ/ARIMA-Model-Stock-Prediction/blob/1256a4e2f18148c27cc2173c9723f6cdecd60bbc/assets/ARIMA%20MODEL.png)

--------------------------

Deployed Web APP
https://bbhuwanj-arima.streamlit.app/ 


--------------------------

## 🚀 Features

- 📂 Upload your own stock CSV file or use a sample dataset
- 🧪 Perform ADF test to check for stationarity
- ⚙️ ARIMA(2,0,4) model fitted to historical data
- 📉 Forecast future stock prices (next 30 days)
- 📊 Evaluate model performance using RMSE, accuracy
- 📈 Visualize residuals, ACF & PACF plots

----------------------------

## 🛠️ Installation


# Clone the repository
```bash
git clone https://github.com/BBhuwanJ/ARIMA-Model-Stock-Prediction.git

```bash
cd lstm-model

# Install dependencies
```bash
pip install -r requirements.txt

# Run the Model
```bash
streamlit run ARIMA_Model.py
