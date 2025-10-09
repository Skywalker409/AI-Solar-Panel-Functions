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

import numpy as np
import joblib
import subprocess
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from colorama import Fore, Style

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
    """Use DC model to predict Max Voltage"""
    X = np.array([[vout, iout, irradiance, temperature]], dtype=np.float32)
    y_pred = dc_model.predict(X, verbose=0)
    return float(np.mean(y_pred))

def predict_max_power(temperature, irradiance):
    """Use Prediction model to predict Max Power"""
    X_input = np.array([[temperature, irradiance]], dtype=np.float32)
    X_scaled = scaler_X.transform(X_input)
    y_pred_scaled = pred_model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]
    return float(y_pred)

def mech_ai_control():
    """Run Mechanical AI on sensor inputs and send movement to solar_tracker.py"""
    try:
        # Get sensor inputs
        i1 = float(input("Enter sensor current i1: "))
        i2 = float(input("Enter sensor current i2: "))
        i3 = float(input("Enter sensor current i3: "))
        i4 = float(input("Enter sensor current i4: "))

        # Run prediction
        X = np.array([[i1, i2, i3, i4]], dtype=np.float32)
        X_s = mech_x_scaler.transform(X)
        y_pred_s = mech_model.predict(X_s, verbose=0)
        dtheta_h, dtheta_v = mech_y_scaler.inverse_transform(y_pred_s)[0]

        printc(f"🔮 MECH-AI Predicted Move: dtheta_h={dtheta_h:.3f}, dtheta_v={dtheta_v:.3f}", "green")

        # Confirm action
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
# Interactive Menu
# ==================================================
if __name__ == "__main__":
    printc("\n=== AI Model Interaction Console ===", "yellow")
    printc("1. DC Model (Predict Max Voltage)", "green")
    printc("2. Prediction Model (Predict Max Power)", "green")
    printc("3. Mechanical AI (Predict + Move Panel)", "green")
    printc("Press Ctrl+C to exit.\n", "red")

    while True:
        try:
            choice = input("Choose model (1, 2, or 3): ").strip()

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

            else:
                printc("⚠️ Invalid choice, please select 1, 2, or 3.\n", "red")

        except KeyboardInterrupt:
            printc("\n👋 Exiting program...", "yellow")
            break
        except Exception as e:
            printc(f"⚠️ Error: {e}\n", "red")
