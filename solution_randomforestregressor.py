# ============================================================
# HAULMARK CHALLENGE — COMPLETE SOLUTION (MEMORY SAFE)
# ============================================================

import numpy as np
import pandas as pd
import os, glob, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings('ignore')

# ⚠️ UPDATE THIS IF RUNNING LOCALLY IN VS CODE
INPUT = '.'  #'/kaggle/input/mindshift-analytics-haul-mark-challenge'
WORK  = '.'   #'/kaggle/working'

print('Files available:')
for f in sorted(os.listdir(INPUT)):
    print(' ', f)

# ── 1. FLEET METADATA ────────────────────────────────────────
print('\n[1/9] Loading fleet...')
fleet = pd.read_csv(f'{INPUT}/fleet.csv')
dumpers = fleet[fleet['fleet'] == 'Dumper'].copy()
dumper_ids = set(dumpers['vehicle'].tolist())
print(f'Dumpers found: {len(dumper_ids)}')

# ── 2 & 3. LOAD ALL TELEMETRY & CLEAN (MEMORY OPTIMIZED) ──────
print('\n[2/9 & 3/9] Loading and Cleaning telemetry...')
tele_files = sorted(glob.glob(f'{INPUT}/telemetry_*.parquet'))
dfs = []

for f in tele_files:
    print(f'  Processing: {os.path.basename(f)}')
    try:
        chunk = pd.read_parquet(f)
        
        # Unify column names across different file formats
        rename_map = {'vehicle_anon': 'vehicle', 'timestamp': 'ts', 'distance': 'cumdist'}
        chunk = chunk.rename(columns={k: v for k, v in rename_map.items() if k in chunk.columns})
        
        # FILTER IMMEDIATELY to save 80% RAM
        chunk = chunk[chunk['vehicle'].isin(dumper_ids)].copy()
        
        # Clean immediately
        chunk['ts'] = pd.to_datetime(chunk['ts'])
        if 'gnss_hdop' in chunk.columns:
            chunk = chunk[~(chunk['gnss_hdop'] > 5)]
        if 'speed' in chunk.columns:
            chunk = chunk[chunk['speed'] <= 80]
            
        dfs.append(chunk)
    except Exception as e:
        print(f"  Skipping {os.path.basename(f)} due to error: {e}")

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(['vehicle', 'ts']).reset_index(drop=True)
print(f'Total clean dumper rows: {len(df):,}')

# ── 4. DAILY BOUNDARY SHIFT (22:00 → next day) ───────────────
print('\n[4/9] Applying 22:00 daily boundary...')
df['report_date'] = (df['ts'] - pd.Timedelta(hours=22)).dt.normalize()

# ── 5. REFUELLING ────────────────────────────────────────────
print('\n[5/9] Loading refuels...')
refuel_file = glob.glob(f'{INPUT}/rfid_refuels*.parquet')[0]
refuels = pd.read_parquet(refuel_file)

r_ts  = 'ts' if 'ts' in refuels.columns else refuels.columns[0]
r_veh = 'vehicle' if 'vehicle' in refuels.columns else 'vehicle_anon'
r_fuel = next((c for c in refuels.columns if any(k in c.lower() for k in ['fuel','qty','quantity','volume','litre','liter']) and c not in [r_ts, r_veh]), refuels.columns[-1])

refuels[r_ts] = pd.to_datetime(refuels[r_ts])
refuels['report_date'] = (refuels[r_ts] - pd.Timedelta(hours=22)).dt.normalize()
refuels = refuels[refuels[r_veh].isin(dumper_ids)]
daily_refuels = refuels.groupby([r_veh, 'report_date'])[r_fuel].sum().reset_index()
daily_refuels.columns = ['vehicle', 'report_date', 'fuel_added']

# ── 6. DAILY FUEL CONSUMED ───────────────────────────────────
print('\n[6/9] Computing fuel consumed...')
df_fuel = df[['vehicle', 'report_date', 'ts', 'fuel_volume']][df['fuel_volume'].notna()].copy()
daily_fuel = df_fuel.groupby(['vehicle', 'report_date']).agg(
    fuel_open  = ('fuel_volume', 'first'),
    fuel_close = ('fuel_volume', 'last'),
).reset_index()

daily_fuel = daily_fuel.merge(daily_refuels, on=['vehicle', 'report_date'], how='left')
daily_fuel['fuel_added']    = daily_fuel['fuel_added'].fillna(0)
daily_fuel['fuel_consumed'] = (daily_fuel['fuel_open'] - daily_fuel['fuel_close'] + daily_fuel['fuel_added']).clip(lower=0)

# ── 7. DUMP CYCLE DETECTION ──────────────────────────────────
print('\n[7/9] Detecting dump events...')
DUMP_THRESHOLD = 2.5
if 'analog_input_1' in df.columns:
    df_dump = df[['vehicle', 'report_date', 'analog_input_1']][df['analog_input_1'].notna()].copy()
    df_dump['is_dumping'] = (df_dump['analog_input_1'] > DUMP_THRESHOLD).astype(int)
    df_dump['prev_dump']  = df_dump.groupby('vehicle')['is_dumping'].shift(1).fillna(0)
    df_dump['dump_event'] = ((df_dump['is_dumping'] == 1) & (df_dump['prev_dump'] == 0)).astype(int)
    daily_dumps = df_dump.groupby(['vehicle', 'report_date']).agg(
        num_dumps      = ('dump_event', 'sum'),
        dump_coverage  = ('analog_input_1', 'count')
    ).reset_index()
else:
    daily_dumps = pd.DataFrame(columns=['vehicle', 'report_date', 'num_dumps', 'dump_coverage'])

# ── 8. FEATURE ENGINEERING ───────────────────────────────────
print('\n[8/9] Engineering features (may take ~1 min)...')

for col in ['altitude', 'speed', 'ignition', 'cumdist', 'disthav', 'angle']:
    if col not in df.columns: df[col] = 0

features = df.groupby(['vehicle', 'report_date']).agg(
    altitude_gain_m = ('altitude', lambda x: x.diff().clip(lower=0).sum()),
    altitude_loss_m = ('altitude', lambda x: x.diff().clip(upper=0).abs().sum()),
    std_altitude    = ('altitude', 'std'),
    altitude_range  = ('altitude', lambda x: x.max() - x.min()),
    mean_speed      = ('speed',    'mean'),
    max_speed       = ('speed',    'max'),
    std_speed       = ('speed',    'std'),
    pct_stopped     = ('speed',    lambda x: (x == 0).mean()),
    engine_on_pings = ('ignition', 'sum'),
    total_pings     = ('ignition', 'count'),
    distance_km     = ('cumdist',  lambda x: x.max() - x.min()),
    total_disthav_m = ('disthav',  'sum'),
    std_angle       = ('angle',    'std'),
).reset_index()

features['engine_on_ratio'] = features['engine_on_pings'] / features['total_pings'].clip(lower=1)
features['gain_per_km']     = features['altitude_gain_m'] / features['distance_km'].clip(lower=0.1)

features = features.merge(daily_dumps[['vehicle', 'report_date', 'num_dumps', 'dump_coverage']], on=['vehicle', 'report_date'], how='left')
features['num_dumps']       = features['num_dumps'].fillna(0)
features['has_dump_switch'] = (features['dump_coverage'].fillna(0) > 0).astype(int)

features = features.merge(dumpers[['vehicle', 'tankcap', 'mine_anon']], on='vehicle', how='left')
features['mine_id']    = features['mine_anon'].str.extract(r'(\d+)').astype(float)
features['vehicle_id'] = features['vehicle'].astype('category').cat.codes

# ── 9. TRAIN / PREDICT / SUBMIT ──────────────────────────────
print('\n[9/9] Training model and generating submission...')

FEATURE_COLS = [
    'altitude_gain_m', 'altitude_loss_m', 'std_altitude', 'altitude_range',
    'mean_speed', 'max_speed', 'std_speed', 'pct_stopped',
    'engine_on_pings', 'total_pings', 'engine_on_ratio',
    'distance_km', 'total_disthav_m', 'gain_per_km',
    'num_dumps', 'has_dump_switch',
    'tankcap', 'mine_id', 'vehicle_id', 'std_angle'
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in features.columns]

# Robust Date Normalization
daily_fuel['report_date'] = pd.to_datetime(daily_fuel['report_date']).dt.normalize()
features['report_date']   = pd.to_datetime(features['report_date']).dt.normalize()

full = features.merge(daily_fuel[['vehicle', 'report_date', 'fuel_consumed']], on=['vehicle', 'report_date'], how='left')

# Train on any row that has a valid fuel_consumed value
train = full[full['fuel_consumed'].notna() & (full['fuel_consumed'] > 0)].copy()

print(f"Total rows in X_train: {len(train)}")

if len(train) == 0:
    print("❌ ERROR: Training set is empty. Check Step 6 merge logic.")
else:
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train['fuel_consumed']

    model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=10, 
    random_state=42, 
    n_jobs=-1
    )

    if len(train) > 10:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        print(f'Random Forest CV RMSE: {-cv.mean():.2f} ± {cv.std():.2f} litres')

    model.fit(X_train, y_train)

    # Generate Submission
    id_map = pd.read_csv(f'{INPUT}/id_mapping.csv')
    # .dt.tz_localize(None) removes the timezone to allow the merge
    id_map['report_date'] = pd.to_datetime(id_map['date']).dt.tz_localize(None).dt.normalize()
    
    full['report_date'] = pd.to_datetime(full['report_date']).dt.tz_localize(None).dt.normalize()
    sub = id_map[['id', 'vehicle', 'report_date']].merge(full[FEATURE_COLS + ['vehicle', 'report_date']], on=['vehicle', 'report_date'], how='left')
    sub['predicted_fuel_value'] = model.predict(sub[FEATURE_COLS].fillna(0)).clip(min=0)
    sub[['id', 'predicted_fuel_value']].to_csv(f'{WORK}/submission_randomforestregressor.csv', index=False)

    print('\n✅ submission_randomforestregressor.csv successfully created!')