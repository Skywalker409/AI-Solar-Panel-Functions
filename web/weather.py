import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Coordinates for Texas A&M University, College Station
LAT = 30.6150
LON = -96.3390

# Output file name
OUTPUT_CSV = "weather_forecast_24h.csv"

# Calculate time range: now to 24 hours from now (timezone-naive for comparison)
now = datetime.utcnow()
end_time = now + timedelta(hours=24)

# Format timestamps for the API
start_str = now.strftime("%Y-%m-%dT%H:%M")
end_str = end_time.strftime("%Y-%m-%dT%H:%M")

# Open-Meteo API (free, no key)
url = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,global_tilted_irradiance"
    "&forecast_days=2&timezone=auto"
)

print("Fetching data from Open-Meteo...")
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Extract hourly data
times = data["hourly"]["time"]
temps = data["hourly"]["temperature_2m"]
irrads = data["hourly"].get("global_tilted_irradiance") or [None] * len(times)

# Convert to DataFrame
df = pd.DataFrame({
    "time": pd.to_datetime(times),
    "predicted_irradiance": irrads,
    "predicted_temperature": temps
})

# Keep only the next 24 hours
df = df[(df["time"] >= now) & (df["time"] <= end_time)]

# Resample to 15-minute intervals if not already
df = df.set_index("time").resample("15min").interpolate(method="linear").reset_index()

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_CSV}")
