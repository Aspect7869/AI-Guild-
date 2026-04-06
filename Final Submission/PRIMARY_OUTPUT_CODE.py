# ============================================================
# HAULMARK CHALLENGE — FINAL OPTIMIZED SOLUTION v7
# Strategy:
#   CORE PHYSICS: acons = lph × runhrs (r=0.946 with acons)
#   LPH is stable per vehicle (~46 L/hr median, low variance)
#   STEP 1: Classify whether shift has runhrs > 0 (idle vs active)
#   STEP 2: Predict acons for active shifts using:
#     - Rolling history (strongest signal, autocorr ~0.30)
#     - Physics constants (LPH per vehicle, per shift)
#     - Spatial features from GPKG (haul distance, zone proximity)
#     - Analog_input_1 properly (smart dump counting per mine)
# ============================================================

import numpy as np
import pandas as pd
import os, glob, warnings, gc, sqlite3, struct, math
import lightgbm as lgb
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from scipy.spatial import cKDTree  # 🔥 INJECTED FOR BLAZING FAST SPATIAL MATH 🔥
warnings.filterwarnings('ignore')

INPUT = '.'
WORK  = '.'

# ══════════════════════════════════════════════════════════════
# PART A: SPATIAL UTILITIES (no geopandas needed)
# ══════════════════════════════════════════════════════════════

def wgs84_to_utm45n(lats, lons):
    """Vectorized WGS84 -> UTM Zone 45N conversion."""
    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)
    k0 = 0.9996; a = 6378137.0; e2 = 0.00669438
    lon0 = math.radians(87.0)
    latr = np.radians(lats); lonr = np.radians(lons)
    N = a / np.sqrt(1 - e2 * np.sin(latr)**2)
    T = np.tan(latr)**2
    C = (e2 / (1 - e2)) * np.cos(latr)**2
    A = np.cos(latr) * (lonr - lon0)
    M = a * ((1 - e2/4 - 3*e2**2/64) * latr
             - (3*e2/8 + 3*e2**2/32) * np.sin(2*latr))
    E  = k0 * N * (A + (1-T+C)*A**3/6) + 500000
    Nn = k0 * (M + N * np.tan(latr) * (A**2/2 + (5-T+9*C)*A**4/24))
    return E, Nn

def get_zone_centroids(gpkg_path, table):
    """Read polygon centroids from a GeoPackage table via SQLite."""
    try:
        conn = sqlite3.connect(gpkg_path)
        cur  = conn.cursor()
        cur.execute(f'SELECT geom FROM {table}')
        rows = cur.fetchall()
        conn.close()
        pts = []
        for (blob,) in rows:
            if not blob: continue
            env = (blob[3] >> 1) & 7
            if env >= 1:
                minx, maxx, miny, maxy = struct.unpack_from('<dddd', blob, 8)
                pts.append(((minx+maxx)/2, (miny+maxy)/2))
        return np.array(pts) if pts else np.zeros((0,2))
    except Exception as e:
        print(f'    GPKG read error ({table}): {e}')
        return np.zeros((0,2))

def min_dist_to_zones(utm_x, utm_y, zone_arr):
    """Minimum Euclidean distance using highly optimized cKDTree (Blazing Fast & Memory Safe)."""
    if len(zone_arr) == 0:
        return np.full(len(utm_x), 9999.0)
    
    # 🔥 Build a spatial index (Takes < 1 ms)
    tree = cKDTree(zone_arr)
    
    # 🔥 Stack coordinates into a 2D array
    pts = np.column_stack((utm_x, utm_y))
    
    # 🔥 Query nearest neighbor instantly using all CPU cores
    dists, _ = tree.query(pts, k=1, workers=-1)
    
    return dists

def haul_road_snap_dist(utm_x, utm_y, road_arr):
    """Distance to nearest haul road segment centroid."""
    return min_dist_to_zones(utm_x, utm_y, road_arr)


# ══════════════════════════════════════════════════════════════
# PART B: LOAD SPATIAL DATA
# ══════════════════════════════════════════════════════════════
print('[A] Loading spatial data from GPKG files...')
gpkg_files = sorted(glob.glob(f'{INPUT}/*.gpkg'))
print(f'    Found {len(gpkg_files)} GPKG files')

SPATIAL = {}
for f in gpkg_files:
    # mine_001_anonymized.gpkg -> mine001
    key = os.path.basename(f).replace('_anonymized.gpkg','').replace('_','')
    SPATIAL[key] = {
        'ob_dump':       get_zone_centroids(f, 'ob_dump'),
        'mineral_stock': get_zone_centroids(f, 'mineral_stock'),
        'bench':         get_zone_centroids(f, 'bench'),
        'haul_road':     get_zone_centroids(f, 'haul_road'),
    }
    for layer, arr in SPATIAL[key].items():
        print(f'    {key}/{layer}: {len(arr)} polygons')

# Mine boundary extents for fast point-in-mine check
MINE_BOUNDS = {
    'mine001': (371208, 2398609, 376893, 2403037),  # minX,minY,maxX,maxY UTM
    'mine002': (366978, 2394228, 369122, 2397196),
}


# ══════════════════════════════════════════════════════════════
# PART C: LOAD FLEET & SUMMARY FILES
# ══════════════════════════════════════════════════════════════
print('\n[B] Loading fleet and summary files...')
fleet      = pd.read_csv(f'{INPUT}/fleet.csv')
dumpers    = fleet[fleet['fleet'] == 'Dumper'].copy()
dumper_ids = set(dumpers['vehicle'].tolist())

# Which vehicles have a working dump switch?
has_dump_switch = set(
    dumpers[dumpers.get('dump_switch', pd.Series(0, index=dumpers.index)) == 1]['vehicle'].tolist()
)
# Fallback: mark mine001 dumpers as having switch, mine002 as not
if not has_dump_switch:
    mine1_dumpers = set(dumpers[dumpers['mine_anon'] == 'mine001']['vehicle'])
    has_dump_switch = mine1_dumpers
print(f'    Dumpers with dump switch: {len(has_dump_switch)}')

smry_files = sorted(glob.glob(f'{INPUT}/smry_*_train_ordered.csv'))
print(f'    Summary files: {[os.path.basename(f) for f in smry_files]}')
train_raw = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
train_raw['date']  = pd.to_datetime(train_raw['date'])
train_raw['shift'] = train_raw['shift'].astype(str).str.strip().str.upper()
train_raw['acons'] = pd.to_numeric(train_raw['acons'], errors='coerce').fillna(0).clip(lower=0)
train_raw = train_raw.sort_values(['vehicle','date','shift']).reset_index(drop=True)

print(f'    Train rows: {len(train_raw)} | Vehicles: {train_raw["vehicle"].nunique()}')
print(f'    Date: {train_raw["date"].min().date()} → {train_raw["date"].max().date()}')
print(f'    acons: mean={train_raw["acons"].mean():.1f}  zero={( train_raw["acons"]==0).mean():.1%}')


# ══════════════════════════════════════════════════════════════
# PART D: PHYSICS CONSTANTS PER VEHICLE
# ══════════════════════════════════════════════════════════════
print('\n[C] Computing physics constants (LPH)...')

eng = train_raw[(train_raw['acons'] > 0) & (train_raw['runhrs'] > 0.1)].copy()
eng['lph_raw'] = eng['acons'] / eng['runhrs']
p99 = eng['lph_raw'].quantile(0.99)
eng  = eng[eng['lph_raw'] < p99].copy()

# Per-vehicle LPH (physical engine constant)
veh_lph = eng.groupby('vehicle')['lph_raw'].agg(
    lph_mean='mean', lph_median='median', lph_std='std',
    lph_p25=lambda x: x.quantile(0.25), lph_p75=lambda x: x.quantile(0.75)
).reset_index()

# Per vehicle+shift
vs_lph = eng.groupby(['vehicle','shift'])['lph_raw'].agg(
    lph_vs_median='median', lph_vs_mean='mean', lph_vs_std='std'
).reset_index()

# Per vehicle+shift runhrs distribution
rh_stats = train_raw.groupby(['vehicle','shift'])['runhrs'].agg(
    rh_mean='mean', rh_median='median', rh_std='std',
    rh_p25=lambda x: x.quantile(0.25), rh_p75=lambda x: x.quantile(0.75),
    rh_zero_frac=lambda x: (x==0).mean()
).reset_index()

# Per vehicle+shift acons distribution
ac_stats = train_raw.groupby(['vehicle','shift'])['acons'].agg(
    ac_mean='mean', ac_median='median', ac_std='std',
    ac_zero_frac=lambda x: (x==0).mean()
).reset_index()

# Per mine stats
mine_lph = eng.groupby('mine')['lph_raw'].agg(mine_lph_med='median').reset_index()
print(f'    LPH range: {veh_lph.lph_median.min():.1f}–{veh_lph.lph_median.max():.1f} L/hr')


# ══════════════════════════════════════════════════════════════
# PART E: ROLLING/LAG FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print('\n[D] Building rolling features...')

def build_rolling(df):
    df = df.sort_values('date').copy()
    for col, windows in [('acons', [1,2,3,7,14,30]), ('runhrs', [1,2,3,7,14])]:
        for w in windows:
            if w <= 3:
                df[f'{col}_lag{w}'] = df[col].shift(w)
            else:
                df[f'{col}_roll{w}']= df[col].shift(1).rolling(w, min_periods=1).mean()
                df[f'{col}_std{w}'] = df[col].shift(1).rolling(w, min_periods=2).std().fillna(0)
    # EWM
    df['acons_ewm7']  = df['acons'].shift(1).ewm(span=7,  min_periods=1).mean()
    df['acons_ewm14'] = df['acons'].shift(1).ewm(span=14, min_periods=1).mean()
    df['runhrs_ewm7'] = df['runhrs'].shift(1).ewm(span=7, min_periods=1).mean()
    # Trend: recent vs longer-term average
    df['acons_trend'] = df['acons_roll7'] - df.get('acons_roll14', df['acons_roll7'])
    # Zero-run streak: how many consecutive zero-acons shifts before this
    df['zero_streak'] = df['acons'].shift(1).eq(0).astype(int)
    df['zero_streak'] = df['zero_streak'].groupby(
        (df['acons'].shift(1).ne(0)).cumsum()
    ).cumsum()
    return df

frames = []
for (veh, shift), grp in train_raw.groupby(['vehicle','shift']):
    frames.append(build_rolling(grp))
train_feat = pd.concat(frames, ignore_index=True)

# Merge physics
train_feat = train_feat.merge(veh_lph,   on='vehicle',             how='left')
train_feat = train_feat.merge(vs_lph,    on=['vehicle','shift'],   how='left')
train_feat = train_feat.merge(rh_stats,  on=['vehicle','shift'],   how='left')
train_feat = train_feat.merge(ac_stats,  on=['vehicle','shift'],   how='left')
train_feat = train_feat.merge(
    dumpers[['vehicle','tankcap','mine_anon']], on='vehicle', how='left'
)
train_feat = train_feat.merge(
    mine_lph, left_on='mine', right_on='mine', how='left'
)

# Encode
train_feat['mine_id']    = train_feat['mine_anon'].str.extract(r'(\d+)').astype(float).fillna(0)
train_feat['vehicle_id'] = train_feat['vehicle'].astype('category').cat.codes
train_feat['shift_id']   = train_feat['shift'].map({'A':0,'B':1,'C':2})
train_feat['day_of_week']= train_feat['date'].dt.dayofweek
train_feat['day_of_month']= train_feat['date'].dt.day
train_feat['month']      = train_feat['date'].dt.month
train_feat['is_weekend'] = (train_feat['day_of_week'] >= 5).astype(int)

# Physics-derived features
train_feat['expected_acons']     = train_feat['lph_vs_median'] * train_feat['rh_median']
train_feat['expected_acons_ewm'] = train_feat['lph_vs_median'] * train_feat['runhrs_ewm7']
train_feat['acons_vs_median']    = train_feat['acons_roll7'] / (train_feat['ac_median'] + 1)
train_feat['runhrs_vs_median']   = train_feat['runhrs_roll7'] / (train_feat['rh_median'] + 0.01)
train_feat['has_dump_switch_flag'] = train_feat['vehicle'].isin(has_dump_switch).astype(int)

print(f'    Feature rows: {len(train_feat)}')


# ══════════════════════════════════════════════════════════════
# PART F: TELEMETRY + SPATIAL FEATURES
# ══════════════════════════════════════════════════════════════
print('\n[E] Processing telemetry with spatial features...')

WANT = [
    'vehicle','ts','latitude','longitude','altitude','speed',
    'ignition','angle','analog_input_1','disthav','cumdist',
    'fuel_volume','shift_dpr','date_dpr','total_trip','km_dpr',
    'tonnage','mine_anon'
]

tele_files = sorted(glob.glob(f'{INPUT}/telemetry*.parquet'))
tele_summaries = []

for f in tele_files:
    print(f'    {os.path.basename(f)}')
    available = pq.read_schema(f).names
    load_cols = [c for c in WANT if c in available]
    chunk = pd.read_parquet(f, columns=load_cols)
    if 'fuel_volume' not in chunk.columns: chunk['fuel_volume'] = np.nan

    chunk = chunk[chunk['vehicle'].isin(dumper_ids)].copy()
    if chunk.empty: continue

    chunk['ts'] = pd.to_datetime(chunk['ts'])
    chunk = chunk.sort_values(['vehicle','ts'])
    chunk = chunk[chunk['speed'] <= 80]

    # Shift assignment
    if 'date_dpr' in chunk.columns and 'shift_dpr' in chunk.columns:
        chunk['report_date'] = pd.to_datetime(chunk['date_dpr']).dt.strftime('%Y-%m-%d')
        chunk['report_shift'] = chunk['shift_dpr'].astype(str).str.strip().str.upper()
    else:
        chunk['report_date'] = (chunk['ts'] - pd.Timedelta(hours=22)).dt.strftime('%Y-%m-%d')
        h = chunk['ts'].dt.hour
        chunk['report_shift'] = np.where(h < 6, 'C', np.where(h < 14, 'A', 'B'))

    # ── SPATIAL FEATURES ────────────────────────────────────
    lat_col = next((c for c in chunk.columns if c.lower() in ['latitude','lat']), None)
    lon_col = next((c for c in chunk.columns if c.lower() in ['longitude','lon']), None)

    chunk['dist_to_dump']  = 9999.0
    chunk['dist_to_load']  = 9999.0
    chunk['dist_to_bench'] = 9999.0
    chunk['dist_to_road']  = 9999.0
    chunk['at_dump_zone']  = 0
    chunk['at_load_zone']  = 0

    if lat_col and lon_col:
        has_gps = chunk[[lat_col, lon_col]].notna().all(axis=1)
        if has_gps.any():
            sub_gps = chunk[has_gps]
            utm_x, utm_y = wgs84_to_utm45n(sub_gps[lat_col].values, sub_gps[lon_col].values)

            for mine_key, sp in SPATIAL.items():
                mine_num = mine_key.replace('mine', '')
                in_mine  = chunk['mine_anon'].str.endswith(mine_num).fillna(False)
                gps_and_mine = has_gps & in_mine

                if gps_and_mine.sum() == 0:
                    continue

                sub_m = chunk[gps_and_mine]
                mx, my = wgs84_to_utm45n(sub_m[lat_col].values, sub_m[lon_col].values)

                d_dump  = min_dist_to_zones(mx, my, sp['ob_dump'])
                d_load  = min_dist_to_zones(mx, my, sp['mineral_stock'])
                d_bench = min_dist_to_zones(mx, my, sp['bench'])
                d_road  = min_dist_to_zones(mx, my, sp['haul_road'])

                chunk.loc[gps_and_mine, 'dist_to_dump']  = d_dump
                chunk.loc[gps_and_mine, 'dist_to_load']  = d_load
                chunk.loc[gps_and_mine, 'dist_to_bench'] = d_bench
                chunk.loc[gps_and_mine, 'dist_to_road']  = d_road
                # Proximity flags (within 300m = in zone)
                chunk.loc[gps_and_mine, 'at_dump_zone'] = (d_dump  < 200).astype(int)
                chunk.loc[gps_and_mine, 'at_load_zone'] = (d_load  < 200).astype(int)

    # ── DUMP SWITCH (smart — per mine) ──────────────────────
    chunk['dump_event'] = 0

    if 'analog_input_1' in chunk.columns:
        has_sw = chunk['vehicle'].isin(has_dump_switch)
        ai_valid = chunk['analog_input_1'].notna() & has_sw
        if ai_valid.any():
            is_dump = (chunk['analog_input_1'] > 2.5).astype(int)
            prev_d  = chunk.groupby('vehicle')['analog_input_1'].shift(1).fillna(0)
            prev_dump = (prev_d > 2.5).astype(int)
            chunk.loc[ai_valid, 'dump_event'] = (
                (is_dump[ai_valid] == 1) & (prev_dump[ai_valid] == 0)
            ).astype(int)

    mine2_mask = chunk['mine_anon'] == 'mine002'
    if mine2_mask.any() and 'at_dump_zone' in chunk.columns:
        prev_at_dump = chunk.groupby('vehicle')['at_dump_zone'].shift(1).fillna(0)
        spatial_dump = ((chunk['at_dump_zone'] == 1) & (prev_at_dump == 0))
        chunk.loc[mine2_mask & spatial_dump, 'dump_event'] = 1

    # ── ALTITUDE FEATURES ───────────────────────────────────
    chunk['alt_diff'] = chunk.groupby('vehicle')['altitude'].diff()
    chunk['alt_gain'] = chunk['alt_diff'].clip(lower=0)
    chunk['alt_loss'] = chunk['alt_diff'].clip(upper=0).abs()
    chunk['is_idle']  = ((chunk['ignition'] == 1) & (chunk['speed'] == 0)).astype(int)

    # ── DAILY SHIFT AGGREGATION ─────────────────────────────
    agg = {
        'altitude_gain_m': ('alt_gain',      'sum'),
        'altitude_loss_m': ('alt_loss',      'sum'),
        'std_altitude':    ('altitude',      'std'),
        'net_lift':        ('altitude',      lambda x: float(x.iloc[-1]) - float(x.iloc[0])),
        'mean_speed':      ('speed',         'mean'),
        'max_speed':       ('speed',         'max'),
        'std_speed':       ('speed',         'std'),
        'pct_stopped':     ('speed',         lambda x: (x==0).mean()),
        'engine_on_pings': ('ignition',      'sum'),
        'idle_pings':      ('is_idle',       'sum'),
        'total_pings':     ('ignition',      'count'),
        'distance_km':     ('cumdist',       lambda x: float(x.max())-float(x.min())),
        'disthav_sum':     ('disthav',       'sum'),
        'num_dumps':       ('dump_event',    'sum'),
        'fuel_open':       ('fuel_volume',   'first'),
        'fuel_close':      ('fuel_volume',   'last'),
        'mean_dist_dump':  ('dist_to_dump',  'mean'),
        'min_dist_dump':   ('dist_to_dump',  'min'),
        'mean_dist_load':  ('dist_to_load',  'mean'),
        'min_dist_load':   ('dist_to_load',  'min'),
        'mean_dist_road':  ('dist_to_road',  'mean'),
        'pct_at_dump':     ('at_dump_zone',  'mean'),
        'pct_at_load':     ('at_load_zone',  'mean'),
        'total_at_dump':   ('at_dump_zone',  'sum'),
    }
    if 'total_trip' in chunk.columns:
        agg['trips_dpr'] = ('total_trip', 'max')
    if 'tonnage' in chunk.columns:
        agg['avg_tonnage'] = ('tonnage', 'mean')
    if 'km_dpr' in chunk.columns:
        agg['km_dpr_max'] = ('km_dpr', 'max')

    day_shift = chunk.groupby(['vehicle','report_date','report_shift']).agg(**agg).reset_index()
    day_shift.columns = ['vehicle','date','shift'] + list(day_shift.columns[3:])
    day_shift['date'] = pd.to_datetime(day_shift['date'])
    day_shift['shift'] = day_shift['shift'].astype(str).str.strip().str.upper()

    tele_summaries.append(day_shift)
    del chunk; gc.collect()

tele_feat = pd.concat(tele_summaries, ignore_index=True)
del tele_summaries; gc.collect()

tele_feat['engine_on_ratio']    = tele_feat['engine_on_pings'] / tele_feat['total_pings'].clip(lower=1)
tele_feat['idle_ratio']         = tele_feat['idle_pings'] / tele_feat['engine_on_pings'].clip(lower=1)
tele_feat['gain_per_km']        = tele_feat['altitude_gain_m'] / tele_feat['distance_km'].clip(lower=0.1)
tele_feat['fuel_consumed_tele'] = (tele_feat['fuel_open'] - tele_feat['fuel_close']).clip(lower=0)
tele_feat['runhrs_est']         = tele_feat['engine_on_pings'] * 20 / 3600  # ~20s per ping
tele_feat['dumps_per_hr']       = tele_feat['num_dumps'] / (tele_feat['runhrs_est'] + 0.1)
tele_feat['haul_difficulty']    = tele_feat['mean_dist_dump'] / (tele_feat['mean_dist_load'] + 1)

print(f'    Telemetry feature rows: {len(tele_feat):,}')

# ── MERGE TELE INTO TRAIN ────────────────────────────────────
print('\n[F] Merging telemetry into training features...')
train_merged = train_feat.merge(tele_feat, on=['vehicle','date','shift'], how='left')
match_rate = train_merged['engine_on_pings'].notna().mean()
print(f'    Tele match rate: {match_rate:.1%} of training rows')


# ══════════════════════════════════════════════════════════════
# PART G: DEFINE FEATURE COLUMNS
# ══════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'acons_lag1','acons_lag2','acons_lag3',
    'acons_roll7','acons_roll14','acons_roll30',
    'acons_ewm7','acons_ewm14',
    'acons_std7','acons_std14',
    'acons_trend',
    'runhrs_lag1','runhrs_lag2','runhrs_lag3',
    'runhrs_roll7','runhrs_roll14',
    'runhrs_ewm7',
    'zero_streak',
    'lph_mean','lph_median','lph_std',
    'lph_vs_median','lph_vs_mean','lph_vs_std',
    'rh_mean','rh_median','rh_std','rh_p25','rh_p75','rh_zero_frac',
    'ac_mean','ac_median','ac_std','ac_zero_frac',
    'mine_lph_med',
    'expected_acons','expected_acons_ewm',
    'acons_vs_median','runhrs_vs_median',
    'tankcap','mine_id','vehicle_id','shift_id',
    'has_dump_switch_flag',
    'day_of_week','day_of_month','month','is_weekend',
    'altitude_gain_m','altitude_loss_m','std_altitude',
    'net_lift','gain_per_km',
    'mean_speed','max_speed','std_speed','pct_stopped',
    'engine_on_pings','idle_pings','engine_on_ratio','idle_ratio',
    'runhrs_est',
    'distance_km','disthav_sum',
    'num_dumps','dumps_per_hr',
    'mean_dist_dump','min_dist_dump',
    'mean_dist_load','min_dist_load',
    'mean_dist_road',
    'pct_at_dump','pct_at_load','total_at_dump',
    'haul_difficulty',
    'fuel_consumed_tele',
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in train_merged.columns]
print(f'\n[G] Using {len(FEATURE_COLS)} features')


# ══════════════════════════════════════════════════════════════
# PART H: TWO-STAGE MODEL
# ══════════════════════════════════════════════════════════════
print('\n[H] Training two-stage model...')

train_model = train_merged[train_merged['acons_lag1'].notna()].copy()
X = train_model[FEATURE_COLS].fillna(0)
y = train_model['acons']
y_binary = (y > 0).astype(int)
groups = train_model['vehicle'].astype('category').cat.codes

lgb_params_cls = dict(
    n_estimators=1000, learning_rate=0.03, max_depth=6,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
    objective='binary', metric='binary_logloss'
)
lgb_params_reg = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=7,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
)

clf = lgb.LGBMClassifier(**lgb_params_cls)
reg = lgb.LGBMRegressor(**lgb_params_reg)

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(X))
oof_proba = np.zeros(len(X))

print('    Cross-validation...')
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    yb_tr = y_binary.iloc[tr_idx]

    clf_f = lgb.LGBMClassifier(**lgb_params_cls)
    clf_f.fit(X_tr, yb_tr,
              eval_set=[(X_val, y_binary.iloc[val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
    proba = clf_f.predict_proba(X_val)[:,1]

    active_tr = yb_tr == 1
    reg_f = lgb.LGBMRegressor(**lgb_params_reg)
    reg_f.fit(X_tr[active_tr], y_tr[active_tr],
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    reg_preds = reg_f.predict(X_val).clip(min=0)

    oof_preds[val_idx] = np.where(proba > 0.3, reg_preds, 0.0)
    oof_proba[val_idx] = proba

    fold_rmse = np.sqrt(((y.iloc[val_idx].values - oof_preds[val_idx])**2).mean())
    print(f'    Fold {fold+1} RMSE: {fold_rmse:.2f}L')

oof_rmse = np.sqrt(((y.values - oof_preds)**2).mean())
print(f'\n    ★ OOF RMSE: {oof_rmse:.2f}L ← expected leaderboard score')

print('\n    Retraining on full data...')
clf.fit(X, y_binary, callbacks=[lgb.log_evaluation(-1)])
active_mask = y_binary == 1
reg.fit(X[active_mask], y[active_mask], callbacks=[lgb.log_evaluation(-1)])


# ══════════════════════════════════════════════════════════════
# PART I: BUILD TEST FEATURES
# ══════════════════════════════════════════════════════════════
print('\n[I] Building test features...')
id_map = pd.read_csv(f'{INPUT}/id_mapping_new.csv')
id_map['date']  = pd.to_datetime(id_map['date'])
id_map['shift'] = id_map['shift'].astype(str).str.strip().str.upper()

test_rows = []
for _, row in id_map.iterrows():
    veh, date, shift = row['vehicle'], row['date'], row['shift']
    hist = train_raw[
        (train_raw['vehicle'] == veh) &
        (train_raw['shift']   == shift) &
        (train_raw['date']    < date)
    ].sort_values('date')

    if len(hist) == 0:
        test_rows.append({'vehicle':veh,'date':date,'shift':shift})
        continue

    a = hist['acons'].values
    r = hist['runhrs'].values

    feat = {
        'vehicle': veh, 'date': date, 'shift': shift,
        'acons_lag1':  a[-1]  if len(a)>=1 else np.nan,
        'acons_lag2':  a[-2]  if len(a)>=2 else np.nan,
        'acons_lag3':  a[-3]  if len(a)>=3 else np.nan,
        'runhrs_lag1': r[-1]  if len(r)>=1 else np.nan,
        'runhrs_lag2': r[-2]  if len(r)>=2 else np.nan,
        'runhrs_lag3': r[-3]  if len(r)>=3 else np.nan,
        'acons_roll7':   np.mean(a[-7:]),
        'acons_roll14':  np.mean(a[-14:]),
        'acons_roll30':  np.mean(a[-30:]),
        'acons_std7':    np.std(a[-7:]) if len(a[-7:])>1 else 0,
        'acons_std14':   np.std(a[-14:]) if len(a[-14:])>1 else 0,
        'runhrs_roll7':  np.mean(r[-7:]),
        'runhrs_roll14': np.mean(r[-14:]),
        'acons_ewm7':   pd.Series(a).ewm(span=7, min_periods=1).mean().iloc[-1],
        'acons_ewm14':  pd.Series(a).ewm(span=14, min_periods=1).mean().iloc[-1],
        'runhrs_ewm7':  pd.Series(r).ewm(span=7, min_periods=1).mean().iloc[-1],
        'acons_trend':  np.mean(a[-7:]) - np.mean(a[-14:]) if len(a)>=7 else 0,
        'zero_streak':  int(np.sum(np.cumprod(np.array(list(reversed(list(a == 0))), dtype=float)))),
    }
    test_rows.append(feat)

test_feat = pd.DataFrame(test_rows)
test_feat['date']  = pd.to_datetime(test_feat['date'])
test_feat['shift'] = test_feat['shift'].astype(str).str.strip().str.upper()

test_feat = test_feat.merge(veh_lph,  on='vehicle',          how='left')
test_feat = test_feat.merge(vs_lph,   on=['vehicle','shift'], how='left')
test_feat = test_feat.merge(rh_stats, on=['vehicle','shift'], how='left')
test_feat = test_feat.merge(ac_stats, on=['vehicle','shift'], how='left')
test_feat = test_feat.merge(dumpers[['vehicle','tankcap','mine_anon']], on='vehicle', how='left')
test_feat = test_feat.merge(mine_lph, left_on='mine_anon', right_on='mine', how='left')
test_feat['mine_id']    = test_feat['mine_anon'].str.extract(r'(\d+)').astype(float).fillna(0)
test_feat['vehicle_id'] = test_feat['vehicle'].astype('category').cat.codes
test_feat['shift_id']   = test_feat['shift'].map({'A':0,'B':1,'C':2})
test_feat['day_of_week'] = test_feat['date'].dt.dayofweek
test_feat['day_of_month']= test_feat['date'].dt.day
test_feat['month']       = test_feat['date'].dt.month
test_feat['is_weekend']  = (test_feat['day_of_week'] >= 5).astype(int)
test_feat['expected_acons']     = test_feat['lph_vs_median'] * test_feat['rh_median']
test_feat['expected_acons_ewm'] = test_feat['lph_vs_median'] * test_feat['runhrs_ewm7'].fillna(test_feat['rh_median'])
test_feat['acons_vs_median']    = test_feat['acons_roll7'] / (test_feat['ac_median'] + 1)
test_feat['runhrs_vs_median']   = test_feat['runhrs_roll7'] / (test_feat['rh_median'] + 0.01)
test_feat['has_dump_switch_flag'] = test_feat['vehicle'].isin(has_dump_switch).astype(int)

test_feat = test_feat.merge(tele_feat, on=['vehicle','date','shift'], how='left')

if 'engine_on_pings' in test_feat.columns:
    test_feat['engine_on_ratio']    = test_feat['engine_on_pings'] / test_feat['total_pings'].clip(lower=1)
    test_feat['idle_ratio']         = test_feat['idle_pings'] / test_feat['engine_on_pings'].clip(lower=1)
    test_feat['gain_per_km']        = test_feat['altitude_gain_m'] / test_feat['distance_km'].clip(lower=0.1)
    test_feat['fuel_consumed_tele'] = (test_feat['fuel_open'] - test_feat['fuel_close']).clip(lower=0)
    test_feat['runhrs_est']         = test_feat['engine_on_pings'] * 20 / 3600
    test_feat['dumps_per_hr']       = test_feat['num_dumps'] / (test_feat['runhrs_est'] + 0.1)
    test_feat['haul_difficulty']    = test_feat['mean_dist_dump'] / (test_feat['mean_dist_load'] + 1)

# ══════════════════════════════════════════════════════════════
# PART J: PREDICT & SUBMIT
# ══════════════════════════════════════════════════════════════
print('\n[J] Predicting...')
X_test = test_feat.reindex(columns=FEATURE_COLS).fillna(0)

proba_test = clf.predict_proba(X_test)[:,1]
reg_test   = reg.predict(X_test).clip(min=0)

predictions = np.where(proba_test > 0.30, reg_test, 0.0)

for i in range(len(test_feat)):
    row = test_feat.iloc[i]
    roll14 = row.get('acons_roll14', 1)
    roll30 = row.get('acons_roll30', 1)
    if pd.notna(roll14) and roll14 == 0 and pd.notna(roll30) and roll30 == 0:
        if proba_test[i] < 0.2:
            predictions[i] = 0.0

max_possible = test_feat['lph_p75'].fillna(60) * 8
predictions = np.minimum(predictions, max_possible.values)

sub = id_map.copy()
sub['predicted_fuel_value'] = predictions
sub[['id','predicted_fuel_value']].to_csv(f'{WORK}/submission_new_gbm3.csv', index=False)

print('\n✅ submission_new_gbm3.csv saved!')