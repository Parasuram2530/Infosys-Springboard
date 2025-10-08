# 🌤️ AirAware – Real-time AQI + LSTM Prediction

AirAware is an AI-powered real-time air quality monitoring and forecasting system.  
It integrates **live AQI data** from CPCB API and provides **next-hour + 24h PM2.5 forecasts** using an **LSTM deep learning model**.

---

## ✨ Features
- 📡 **Fetch real-time AQI** from CPCB API (Delhi, Mumbai, Kolkata, Chennai supported)
- 🤖 **LSTM model** trained on historical weather & pollution data
- 🔮 **Next-hour PM2.5 prediction** (dashboard)
- 📊 **24-hour PM2.5 forecast** (training script output)
- 📈 Interactive **Streamlit dashboard** with plots & metrics
- 💾 Saves trained model (`.h5`) and scaler (`.pkl`) for reuse

---


---

## ⚙️ Installation

1. Clone the repo:
```bash
git clone https://github.com/Parasuram2530/Infosys-Springboard.git  

2. Create Environment:
```bash
conda create -n airaware python=3.10 -y
conda activate airaware

pip install -r requirements.txt

3. Training the Model:
python lstm_pm25_forecast_24h.py

4. Running the Dashboard:
streamlit run dashboard.py



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
