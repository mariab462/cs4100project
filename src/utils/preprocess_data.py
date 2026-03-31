import os
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

RAW_DIR = "../../data/raw"  # path to XML files
OUTPUT_CSV = "data/processed/patient_data.csv"

# Helper to parse single-timestamp events
def parse_simple_events(root, tag, value_attr, numeric=True):
    events = []
    element = root.find(tag)
    if element is not None:
        for e in element.findall("event"):
            ts = e.attrib.get("ts") or e.attrib.get("ts_begin")
            if not ts:
                continue
            try:
                ts = pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S")
            except:
                try:
                    ts = pd.to_datetime(ts)
                except:
                    continue
            value = e.attrib.get(value_attr, None)
            if numeric:
                try:
                    value = float(value)
                except:
                    value = None
            else:
                value = value.strip() if value else None
            events.append({"timestamp": ts, tag: value})
    if events:
        return pd.DataFrame(events).set_index("timestamp")
    else:
        return pd.DataFrame(columns=["timestamp", tag])

# Helper to parse range events (ts_begin/ts_end)
def parse_range_events(root, tag, value_attr, numeric=True):
    events = []
    element = root.find(tag)
    if element is not None:
        for e in element.findall("event"):
            ts_begin = e.attrib.get("ts_begin")
            ts_end = e.attrib.get("ts_end") or ts_begin
            if not ts_begin or not ts_end:
                continue
            try:
                ts_begin = pd.to_datetime(ts_begin, format="%d-%m-%Y %H:%M:%S")
            except:
                try:
                    ts_begin = pd.to_datetime(ts_begin)
                except:
                    continue
            try:
                ts_end = pd.to_datetime(ts_end, format="%d-%m-%Y %H:%M:%S")
            except:
                try:
                    ts_end = pd.to_datetime(ts_end)
                except:
                    continue
            if ts_end < ts_begin:
                continue
            value = e.attrib.get(value_attr, None)
            if numeric:
                try:
                    value = float(value)
                except:
                    value = None
            else:
                value = value.strip() if value else None
            idx = pd.date_range(start=ts_begin, end=ts_end, freq="5min")
            for t in idx:
                events.append({"timestamp": t, tag: value})
    if events:
        return pd.DataFrame(events).set_index("timestamp")
    else:
        return pd.DataFrame(columns=["timestamp", tag])

# Bolus (dose + carbs)
def parse_bolus(root):
    events = []
    element = root.find("bolus")
    if element is not None:
        for e in element.findall("event"):
            ts = e.attrib.get("ts_begin") or e.attrib.get("ts_end")
            if not ts:
                continue
            try:
                ts = pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S")
            except:
                try:
                    ts = pd.to_datetime(ts)
                except:
                    continue
            try:
                dose = float(e.attrib.get("dose", 0))
            except:
                dose = None
            try:
                carbs = float(e.attrib.get("bwz_carb_input", 0))
            except:
                carbs = None
            events.append({"timestamp": ts, "bolus": dose, "carbs_from_bolus": carbs})
    if events:
        return pd.DataFrame(events).set_index("timestamp")
    else:
        return pd.DataFrame(columns=["timestamp", "bolus", "carbs_from_bolus"])

# Meals
def parse_meals(root):
    events = []
    element = root.find("meal")
    if element is not None:
        for e in element.findall("event"):
            ts = e.attrib.get("ts")
            if not ts:
                continue
            try:
                ts = pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S")
            except:
                try:
                    ts = pd.to_datetime(ts)
                except:
                    continue
            try:
                carbs = float(e.attrib.get("carbs", 0))
            except:
                carbs = None
            meal_type = e.attrib.get("type", "").strip()
            events.append({"timestamp": ts, "meal_carbs": carbs, "meal_type": meal_type})
    if events:
        return pd.DataFrame(events).set_index("timestamp")
    else:
        return pd.DataFrame(columns=["timestamp", "meal_carbs", "meal_type"])

# Exercise
def parse_exercise(root):
    events = []
    element = root.find("exercise")
    if element is not None:
        for e in element.findall("event"):
            ts = e.attrib.get("ts") or e.attrib.get("ts_begin")
            if not ts:
                continue
            try:
                ts = pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S")
            except:
                try:
                    ts = pd.to_datetime(ts)
                except:
                    continue
            try:
                intensity = float(e.attrib.get("intensity", 0))
            except:
                intensity = None
            try:
                duration = float(e.attrib.get("duration", 0))
            except:
                duration = None
            events.append({"timestamp": ts, "exercise_intensity": intensity, "exercise_duration": duration})
    if events:
        return pd.DataFrame(events).set_index("timestamp")
    else:
        return pd.DataFrame(columns=["timestamp", "exercise_intensity", "exercise_duration"])

# Sleep, work, illness (range events)
def parse_sleep(root):
    return parse_range_events(root, "sleep", "quality").rename(columns={"sleep": "sleep_quality"})

def parse_work(root):
    return parse_range_events(root, "work", "intensity").rename(columns={"work": "work_intensity"})

def parse_illness(root):
    return parse_range_events(root, "illness", "description", numeric=False).rename(columns={"illness": "illness"})

# Main parser for one patient
def parse_patient_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    patient_id = root.attrib.get("id", "unknown")
    try:
        weight = float(root.attrib.get("weight", 0))
    except:
        weight = None

    dfs = []

    # Single timestamp events
    for tag in ["glucose_level", "finger_stick", "basal",
                "basis_heart_rate", "basis_gsr", "basis_skin_temperature",
                "basis_air_temperature", "basis_steps"]:
        df = parse_simple_events(root, tag, "value")
        if not df.empty:
            dfs.append(df)

    # Range events
    temp_basal_df = parse_range_events(root, "temp_basal", "value").rename(columns={"temp_basal": "temp_basal"})
    if not temp_basal_df.empty:
        dfs.append(temp_basal_df)

    # Other events
    for f in [parse_bolus, parse_meals, parse_exercise, parse_sleep, parse_work, parse_illness]:
        df = f(root)
        if not df.empty:
            dfs.append(df)

    # Merge all on timestamp
    if dfs:
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df["patient_id"] = patient_id
        df["weight"] = weight
        df = df.resample("5min").ffill().bfill()
        return df
    else:
        return pd.DataFrame()

# Preprocess all patients
def preprocess_all_patients():
    all_dfs = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".xml"):
            path = os.path.join(RAW_DIR, file)
            df = parse_patient_xml(path)
            if not df.empty:
                all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs).reset_index().rename(columns={"index": "timestamp"})
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = preprocess_all_patients()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Preprocessed data saved to {OUTPUT_CSV}")