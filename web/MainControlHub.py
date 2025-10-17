##########################################################
#               To control Mechanical Part:
# python3 solar_tracker.py home
# python3 solar_tracker.py move-by 2 -1.5
# python3 solar_tracker.py move-to 15 25
# python3 solar_tracker.py return
# python3 solar_tracker.py status
# python3 solar_tracker.py jog az 0.2 --count 10 --delay 0.15
# python3 solar_tracker.py --microsteps 256 --irun 20 --ihold 8 home
# python3 solar_tracker.py --no-uart move-by -0.25 0.25
###########################################################
import os
import numpy as np
import joblib
import subprocess
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from colorama import Fore, Style
from tqdm import tqdm

# --- Utility for colored prints ---
def printc(message, color):
    match color:
        case "green":
            print(Fore.GREEN + message + Style.RESET_ALL)
        case "red":
            print(Fore.RED + message + Style.RESET_ALL)
        case "yellow":
            print(Fore.YELLOW + message + Style.RESET_ALL)
        case _:
            print(message)

# ==================================================
# Load models
# ==================================================
printc("Loading models...", "yellow")
# DC Model (Max Voltage)
dc_model_path = "./AI_Models/DC-DemoModel_2.3.h5"
dc_model = load_model(dc_model_path, compile=False)
dc_model.compile(optimizer="adam", loss=MeanSquaredError())
# Prediction Model (Max Power)
pred_model_path = "./AI_Models/DataV5Model_V3.h5"
pred_model = load_model(pred_model_path, compile=False)
scaler_X = joblib.load("./AI_Models/scaler_X.pkl")
scaler_y = joblib.load("./AI_Models/scaler_y.pkl")
# Mechanical AI Model
mech_model_path = "./AI_Models/mech_ai_mlp.keras"
mech_x_scaler_path = "./AI_Models/x_scaler.joblib"
mech_y_scaler_path = "./AI_Models/y_scaler.joblib"
mech_model = load_model(mech_model_path)
mech_x_scaler = joblib.load(mech_x_scaler_path)
mech_y_scaler = joblib.load(mech_y_scaler_path)
printc("✅ Models loaded successfully!", "green")

# ==================================================
# Inference Functions
# ==================================================
def predict_max_voltage(vout, iout, irradiance, temperature):
    X = np.array([[vout, iout, irradiance, temperature]], dtype=np.float32)
    y_pred = dc_model.predict(X, verbose=0)
    return float(np.mean(y_pred))

def predict_max_power(temperature, irradiance):
    X_input = np.array([[temperature, irradiance]], dtype=np.float32)
    X_scaled = scaler_X.transform(X_input)
    y_pred_scaled = pred_model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]
    return float(y_pred)

def mech_ai_control():
    try:
        i1 = float(input("Enter sensor current i1: "))
        i2 = float(input("Enter sensor current i2: "))
        i3 = float(input("Enter sensor current i3: "))
        i4 = float(input("Enter sensor current i4: "))
        X = np.array([[i1, i2, i3, i4]], dtype=np.float32)
        X_s = mech_x_scaler.transform(X)
        y_pred_s = mech_model.predict(X_s, verbose=0)
        dtheta_h, dtheta_v = mech_y_scaler.inverse_transform(y_pred_s)[0]
        printc(f"🔮 MECH-AI Predicted Move: dtheta_h={dtheta_h:.3f}, dtheta_v={dtheta_v:.3f}", "green")
        confirm = input("Send this move to solar_tracker.py? (y/n): ").strip().lower()
        if confirm == "y":
            cmd = ["python3", "solar_tracker.py", "move-by", str(dtheta_h), str(dtheta_v)]
            printc(f"⚡ Executing: {' '.join(cmd)}", "yellow")
            result = subprocess.run(cmd, capture_output=True, text=True)
            printc(result.stdout, "green")
            if result.stderr:
                printc(result.stderr, "red")
        else:
            printc("❌ Move canceled.", "red")
    except Exception as e:
        printc(f"⚠️ MECH-AI Error: {e}", "red")

# ==================================================
# Weather + Power Prediction
# ==================================================
def predict_daily_power():
    try:
        printc("☀️ Fetching weather forecast from Open-Meteo...", "yellow")
        LAT, LON = 30.6150, -96.3390
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAT}&longitude={LON}"
            "&hourly=temperature_2m,global_tilted_irradiance"
            "&forecast_days=2&timezone=auto"
        )
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()

        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        irrads = data["hourly"].get("global_tilted_irradiance") or [None] * len(times)

        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "temperature": temps,
            "irradiance": irrads
        })

        now = datetime.utcnow()
        midnight = datetime.combine(now.date(), datetime.min.time())
        next_midnight = midnight + timedelta(days=1)
        df = df[(df["time"] >= midnight) & (df["time"] <= next_midnight)]
        df = df.set_index("time").resample("15min").interpolate(method="linear").reset_index()

        printc(f"📈 Running AI prediction over {len(df)} intervals...", "yellow")
        total_energy_Wh = 0.0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting power", unit="step"):
            temp = float(row["temperature"])
            irr = float(row["irradiance"])
            power_W = predict_max_power(temp, irr)
            total_energy_Wh += power_W * 0.25  # 15 minutes = 0.25 hour

        result = {
            "date": now.strftime("%Y-%m-%d"),
            "total_predicted_energy_Wh": round(total_energy_Wh, 2),
            "timestamp_generated": now.isoformat()
        }

        # Write JSON atomically into the same folder as this script (and index.html)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "predicted_daily_power.json")
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=4)
        os.replace(tmp_path, output_path)

        printc(f"✅ Predicted total energy for today: {total_energy_Wh:.2f} Wh", "green")
        printc(f"💾 Saved results to {output_path}", "yellow")
    except Exception as e:
        printc(f"⚠️ Weather Prediction Error: {e}", "red")

# ==================================================
# Interactive Menu
# ==================================================
if __name__ == "__main__":
    printc("\n=== AI Model Interaction Console ===", "yellow")
    printc("1. DC Model (Predict Max Voltage)", "green")
    printc("2. Prediction Model (Predict Max Power)", "green")
    printc("3. Mechanical AI (Predict + Move Panel)", "green")
    printc("4. Predict Total Daily Power (Weather + AI)", "green")
    printc("Press Ctrl+C to exit.\n", "red")
    while True:
        try:
            choice = input("Choose model (1, 2, 3, or 4): ").strip()
            if choice == "1":
                vout = float(input("Enter Vout: "))
                iout = float(input("Enter Iout: "))
                irradiance = float(input("Enter Irradiance (W/m²): "))
                temperature = float(input("Enter Temperature (°C): "))
                prediction = predict_max_voltage(vout, iout, irradiance, temperature)
                printc(f"🔮 Predicted Max Voltage = {prediction:.4f} V\n", "green")
            elif choice == "2":
                temperature = float(input("Enter Temperature (°C): "))
                irradiance = float(input("Enter Irradiance (W/m²): "))
                prediction = predict_max_power(temperature, irradiance)
                printc(f"🔮 Predicted Max Power = {prediction:.4f} W\n", "green")
            elif choice == "3":
                mech_ai_control()
            elif choice == "4":
                predict_daily_power()
            else:
                printc("⚠️ Invalid choice, please select 1–4.\n", "red")
        except KeyboardInterrupt:
            printc("\n👋 Exiting program...", "yellow")
            break
        except Exception as e:
            printc(f"⚠️ Error: {e}\n", "red")