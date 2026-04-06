# ============================================================
# HAULMARK CHALLENGE — SECONDARY OUTPUTS
# Generates all 4 required secondary outputs:
#   1. Route-Level Fuel Benchmark
#   2. Dumper Efficiency Component
#   3. Cycle Segmentation Methodology
#   4. Daily Fuel Consistency
# ============================================================

import numpy as np
import pandas as pd
import os, glob, warnings, gc, sqlite3, struct, math
import lightgbm as lgb
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

INPUT = '.'
WORK  = '.'

# ──────────────────────────────────────────────────────────────
# SHARED HELPERS (same as main solution)
# ──────────────────────────────────────────────────────────────
def wgs84_to_utm45n(lats, lons):
    lats=np.array(lats,float); lons=np.array(lons,float)
    k0=0.9996; a=6378137.0; e2=0.00669438; lon0=math.radians(87.0)
    latr=np.radians(lats); lonr=np.radians(lons)
    N=a/np.sqrt(1-e2*np.sin(latr)**2); T=np.tan(latr)**2
    C=(e2/(1-e2))*np.cos(latr)**2; A=np.cos(latr)*(lonr-lon0)
    M=a*((1-e2/4-3*e2**2/64)*latr-(3*e2/8+3*e2**2/32)*np.sin(2*latr))
    E=k0*N*(A+(1-T+C)*A**3/6)+500000
    Nn=k0*(M+N*np.tan(latr)*(A**2/2+(5-T+9*C)*A**4/24))
    return E, Nn

def get_zone_centroids(gpkg_path, table):
    try:
        conn=sqlite3.connect(gpkg_path); cur=conn.cursor()
        cur.execute(f'SELECT geom FROM {table}'); rows=cur.fetchall(); conn.close()
        pts=[]
        for (blob,) in rows:
            if not blob: continue
            if (blob[3]>>1)&7>=1:
                minx,maxx,miny,maxy=struct.unpack_from('<dddd',blob,8)
                pts.append(((minx+maxx)/2,(miny+maxy)/2))
        return np.array(pts) if pts else np.zeros((0,2))
    except: return np.zeros((0,2))

def min_dist(ux, uy, zones):
    if len(zones)==0: return np.full(len(ux),9999.0)
    from scipy.spatial import cKDTree
    tree=cKDTree(zones)
    d,_=tree.query(np.column_stack([ux,uy]),k=1,workers=-1)
    return d

# ──────────────────────────────────────────────────────────────
# LOAD BASE DATA
# ──────────────────────────────────────────────────────────────
print('Loading base data...')
fleet    = pd.read_csv(f'{INPUT}/fleet.csv')
dumpers  = fleet[fleet['fleet']=='Dumper'].copy()
dumper_ids = set(dumpers['vehicle'].tolist())

if 'dump_switch' in dumpers.columns:
    has_sw = set(dumpers[dumpers['dump_switch']==1]['vehicle'])
else:
    has_sw = set(dumpers[dumpers['mine_anon']=='mine001']['vehicle'])

smry_files = sorted(glob.glob(f'{INPUT}/smry_*_train_ordered.csv'))
train_raw  = pd.concat([pd.read_csv(f) for f in smry_files], ignore_index=True)
train_raw['date']   = pd.to_datetime(train_raw['date'])
train_raw['shift']  = train_raw['shift'].astype(str).str.strip().str.upper()
train_raw['acons']  = pd.to_numeric(train_raw['acons'],  errors='coerce').fillna(0).clip(lower=0)
train_raw['runhrs'] = pd.to_numeric(train_raw['runhrs'], errors='coerce').fillna(0).clip(lower=0)
train_raw = train_raw.sort_values(['vehicle','date','shift']).reset_index(drop=True)

id_map = pd.read_csv(f'{INPUT}/id_mapping_new.csv')
id_map['date']  = pd.to_datetime(id_map['date'])
id_map['shift'] = id_map['shift'].astype(str).str.strip().str.upper()

# Load main submission to get predicted fuel
try:
    main_sub = pd.read_csv(f'{WORK}/submission.csv')
    has_main_preds = True
    print(f'Loaded main submission: {len(main_sub)} rows')
except:
    has_main_preds = False
    print('Warning: main submission not found — using physics baseline for consistency check')

# Mine mapping
mine_veh = train_raw[['vehicle','mine']].drop_duplicates()

# Physics constants
eng = train_raw[(train_raw.acons>0)&(train_raw.runhrs>0.1)].copy()
eng['lph'] = eng['acons']/eng['runhrs']
p99 = eng['lph'].quantile(0.99)
eng = eng[eng.lph < p99]

veh_lph = eng.groupby('vehicle')['lph'].agg(
    lph_med='median', lph_mean='mean', lph_std='std',
    lph_p25=lambda x:x.quantile(0.25), lph_p75=lambda x:x.quantile(0.75)
).reset_index()

vs_lph = eng.groupby(['vehicle','shift'])['lph'].agg(
    lph_vs_med='median', lph_vs_mean='mean'
).reset_index()

rh_stats = train_raw.groupby(['vehicle','shift'])['runhrs'].agg(
    rh_med='median', rh_mean='mean', rh_std='std',
    rh_zero=lambda x:(x==0).mean()
).reset_index()

ac_stats = train_raw.groupby(['vehicle','shift'])['acons'].agg(
    ac_med='median', ac_mean='mean', ac_std='std',
    ac_zero=lambda x:(x==0).mean()
).reset_index()

print(f'Training rows: {len(train_raw)} | Vehicles: {train_raw.vehicle.nunique()}')

# ══════════════════════════════════════════════════════════════
# OUTPUT 1: ROUTE-LEVEL FUEL BENCHMARK
# Definition: Expected fuel consumed per shift for a given route
#             (mine + shift), independent of which dumper operates it.
# Method: Median LPH (stable physical constant) × Median RunHrs
#         for that mine-shift combination across all dumpers.
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('OUTPUT 1: Route-Level Fuel Benchmark')
print('='*60)

route_bm = eng.groupby(['mine','shift']).agg(
    route_lph_med      = ('lph',    'median'),
    route_lph_std      = ('lph',    'std'),
    route_rh_med       = ('runhrs', 'median'),
    route_rh_std       = ('runhrs', 'std'),
    route_acons_med    = ('acons',  'median'),
    route_acons_mean   = ('acons',  'mean'),
    route_acons_std    = ('acons',  'std'),
    n_vehicle_shifts   = ('acons',  'count'),
).reset_index()

# Route benchmark = physics-based: median LPH × median RunHrs
route_bm['route_fuel_benchmark_L'] = (
    route_bm['route_lph_med'] * route_bm['route_rh_med']
).round(2)

# Confidence band: ± 1 std of actual acons
route_bm['benchmark_lower_L'] = (
    route_bm['route_acons_med'] - route_bm['route_acons_std']
).clip(lower=0).round(2)
route_bm['benchmark_upper_L'] = (
    route_bm['route_acons_med'] + route_bm['route_acons_std']
).round(2)

# Rename for clarity
route_bm = route_bm.rename(columns={
    'mine': 'route_mine',
    'shift': 'route_shift',
})

route_bm_out = route_bm[[
    'route_mine','route_shift',
    'route_fuel_benchmark_L','benchmark_lower_L','benchmark_upper_L',
    'route_lph_med','route_rh_med',
    'route_acons_med','route_acons_mean','route_acons_std',
    'n_vehicle_shifts'
]]
route_bm_out.to_csv(f'{WORK}/output1_route_fuel_benchmark.csv', index=False)
print(route_bm_out.to_string())
print(f'\nSaved: output1_route_fuel_benchmark.csv')

# ══════════════════════════════════════════════════════════════
# OUTPUT 2: DUMPER EFFICIENCY COMPONENT
# Definition: How much each dumper's fuel consumption deviates
#             from the route baseline.
# Method:
#   efficiency_ratio = dumper_lph_median / route_lph_median
#   > 1.1 = inefficient (burns >10% more than route average)
#   0.9-1.1 = average efficiency
#   < 0.9 = efficient (burns >10% less than route average)
#   fuel_excess_L = dumper_acons_median - route_fuel_benchmark
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('OUTPUT 2: Dumper Efficiency Component')
print('='*60)

# Per vehicle+shift efficiency
veh_eff = eng.groupby(['vehicle','shift']).agg(
    veh_lph_med   = ('lph',    'median'),
    veh_lph_std   = ('lph',    'std'),
    veh_rh_med    = ('runhrs', 'median'),
    veh_acons_med = ('acons',  'median'),
    n_shifts      = ('acons',  'count'),
).reset_index()

# Merge mine info
veh_eff = veh_eff.merge(mine_veh, on='vehicle', how='left')
veh_eff = veh_eff.merge(
    route_bm[['route_mine','route_shift','route_lph_med','route_rh_med','route_fuel_benchmark_L']],
    left_on=['mine','shift'], right_on=['route_mine','route_shift'], how='left'
).drop(columns=['route_mine','route_shift'])

# Efficiency metrics
veh_eff['lph_efficiency_ratio'] = (
    veh_eff['veh_lph_med'] / veh_eff['route_lph_med']
).round(4)

veh_eff['rh_utilisation_ratio'] = (
    veh_eff['veh_rh_med'] / veh_eff['route_rh_med']
).round(4)

veh_eff['fuel_excess_vs_route_L'] = (
    veh_eff['veh_acons_med'] - veh_eff['route_fuel_benchmark_L']
).round(2)

veh_eff['efficiency_label'] = pd.cut(
    veh_eff['lph_efficiency_ratio'],
    bins=[0, 0.90, 1.10, 999],
    labels=['efficient', 'average', 'inefficient']
).astype(str)

# Overall per-vehicle summary (average across shifts)
veh_summary = veh_eff.groupby('vehicle').agg(
    mean_lph_eff_ratio    = ('lph_efficiency_ratio', 'mean'),
    mean_rh_util_ratio    = ('rh_utilisation_ratio',  'mean'),
    mean_fuel_excess_L    = ('fuel_excess_vs_route_L','mean'),
    dominant_eff_label    = ('efficiency_label', lambda x: x.mode()[0]),
    mine                  = ('mine', 'first'),
).round(4).reset_index()

veh_summary['overall_efficiency_label'] = pd.cut(
    veh_summary['mean_lph_eff_ratio'],
    bins=[0, 0.90, 1.10, 999],
    labels=['efficient', 'average', 'inefficient']
).astype(str)

# Merge with tank capacity
veh_summary = veh_summary.merge(
    dumpers[['vehicle','tankcap']], on='vehicle', how='left'
)

# Sort: most inefficient first (highest excess fuel)
veh_summary = veh_summary.sort_values('mean_fuel_excess_L', ascending=False)

# Save detailed (per vehicle+shift)
veh_eff_out = veh_eff[[
    'vehicle','mine','shift',
    'veh_lph_med','route_lph_med','lph_efficiency_ratio',
    'veh_rh_med','route_rh_med','rh_utilisation_ratio',
    'veh_acons_med','route_fuel_benchmark_L','fuel_excess_vs_route_L',
    'efficiency_label','n_shifts'
]].sort_values(['vehicle','shift'])

veh_eff_out.to_csv(f'{WORK}/output2_dumper_efficiency_by_shift.csv', index=False)
veh_summary.to_csv(f'{WORK}/output2_dumper_efficiency_summary.csv', index=False)

print('Per-vehicle summary:')
print(veh_summary[[
    'vehicle','mine','mean_lph_eff_ratio','mean_fuel_excess_L',
    'overall_efficiency_label'
]].to_string())
print(f'\nSaved: output2_dumper_efficiency_by_shift.csv')
print(f'Saved: output2_dumper_efficiency_summary.csv')

# ══════════════════════════════════════════════════════════════
# OUTPUT 3: CYCLE SEGMENTATION METHODOLOGY
# Definition: How many haul cycles (Load→Travel→Dump) each dumper
#             completed per shift. Validated against total_trip from DPR.
# Method:
#   Mine001: analog_input_1 > 2.5V rising edge = dump event
#   Mine002: spatial transition into ob_dump zone = dump event
#   Fuel per cycle = acons / num_cycles (when cycles > 0)
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('OUTPUT 3: Cycle Segmentation from Telemetry')
print('='*60)

# Load spatial data
SPATIAL = {}
for f in sorted(glob.glob(f'{INPUT}/*.gpkg')):
    key = os.path.basename(f).replace('_anonymized.gpkg','').replace('_','')
    SPATIAL[key] = {
        'ob_dump':       get_zone_centroids(f, 'ob_dump'),
        'mineral_stock': get_zone_centroids(f, 'mineral_stock'),
    }

WANT_TELE = [
    'vehicle','ts','latitude','longitude',
    'analog_input_1','shift_dpr','date_dpr','total_trip','mine_anon'
]

tele_files = sorted(glob.glob(f'{INPUT}/telemetry*.parquet'))
cycle_parts = []

for f in tele_files:
    avail = pq.read_schema(f).names
    cols  = [c for c in WANT_TELE if c in avail]
    chunk = pd.read_parquet(f, columns=cols)
    chunk = chunk[chunk['vehicle'].isin(dumper_ids)].copy()
    if chunk.empty: continue

    chunk['ts'] = pd.to_datetime(chunk['ts'])
    chunk = chunk.sort_values(['vehicle','ts'])

    # Shift assignment
    if 'date_dpr' in chunk.columns and 'shift_dpr' in chunk.columns:
        chunk['rdate']  = pd.to_datetime(chunk['date_dpr']).dt.strftime('%Y-%m-%d')
        chunk['rshift'] = chunk['shift_dpr'].astype(str).str.strip().str.upper()
    else:
        chunk['rdate']  = (chunk['ts']-pd.Timedelta(hours=22)).dt.strftime('%Y-%m-%d')
        h = chunk['ts'].dt.hour
        chunk['rshift'] = np.where(h<6,'C',np.where(h<14,'A','B'))

    # Method A: analog_input_1 rising edge (mine001 vehicles with dump switch)
    chunk['dump_ev_ai'] = 0
    if 'analog_input_1' in chunk.columns:
        sw_mask = chunk['vehicle'].isin(has_sw) & chunk['analog_input_1'].notna()
        if sw_mask.any():
            is_d   = (chunk['analog_input_1'] > 2.5).astype(int)
            prev_d = chunk.groupby('vehicle')['analog_input_1'].shift(1).fillna(0)
            chunk.loc[sw_mask, 'dump_ev_ai'] = (
                (is_d[sw_mask]==1) & (prev_d[sw_mask]<=2.5)
            ).astype(int)

    # Method B: spatial zone transition (mine002 — analog unreliable)
    lat_col = next((c for c in chunk.columns if c.lower() in ['latitude','lat']),None)
    lon_col = next((c for c in chunk.columns if c.lower() in ['longitude','lon']),None)
    chunk['dump_ev_spatial'] = 0

    if lat_col and lon_col:
        has_gps = chunk[[lat_col,lon_col]].notna().all(axis=1)
        for mkey, sp in SPATIAL.items():
            mnum = mkey.replace('mine','')
            mask = has_gps & chunk['mine_anon'].str.endswith(mnum).fillna(False)
            if mask.sum()==0 or len(sp['ob_dump'])==0: continue
            sub = chunk[mask]
            ux,uy = wgs84_to_utm45n(sub[lat_col].values, sub[lon_col].values)
            d_dump = min_dist(ux,uy,sp['ob_dump'])
            at_dump = (d_dump < 200).astype(int)
            chunk.loc[mask,'at_dump_zone'] = at_dump

        if 'at_dump_zone' in chunk.columns:
            prev_at = chunk.groupby('vehicle')['at_dump_zone'].shift(1).fillna(0)
            m2 = chunk['mine_anon']=='mine002'
            chunk.loc[m2, 'dump_ev_spatial'] = (
                (chunk.loc[m2,'at_dump_zone']==1) & (prev_at[m2]==0)
            ).astype(int)

    # Combined dump event: AI for mine001, spatial for mine002
    chunk['dump_event'] = chunk['dump_ev_ai']
    m2 = chunk['mine_anon']=='mine002'
    chunk.loc[m2, 'dump_event'] = chunk.loc[m2, 'dump_ev_spatial']

    # Aggregate per shift
    agg_dict = {
        'n_dumps_ai':      ('dump_ev_ai',      'sum'),
        'n_dumps_spatial': ('dump_ev_spatial',  'sum'),
        'n_dumps_combined':('dump_event',       'sum'),
        'n_pings':         ('vehicle',          'count'),
    }
    if 'total_trip' in chunk.columns:
        agg_dict['total_trip_dpr'] = ('total_trip', 'max')

    ds = chunk.groupby(['vehicle','rdate','rshift']).agg(**agg_dict).reset_index()
    ds.columns = ['vehicle','date','shift'] + list(ds.columns[3:])
    ds['date'] = pd.to_datetime(ds['date'])
    ds['shift']= ds['shift'].astype(str).str.strip().str.upper()
    cycle_parts.append(ds)
    del chunk; gc.collect()

cycle_df = pd.concat(cycle_parts, ignore_index=True)

# Merge with training summary to get actual acons and runhrs
cycle_df = cycle_df.merge(
    train_raw[['vehicle','date','shift','acons','runhrs']],
    on=['vehicle','date','shift'], how='left'
)
cycle_df = cycle_df.merge(mine_veh, on='vehicle', how='left')

# Compute fuel per cycle
cycle_df['fuel_per_cycle_L'] = np.where(
    cycle_df['n_dumps_combined'] > 0,
    cycle_df['acons'] / cycle_df['n_dumps_combined'],
    np.nan
)

# Validate: compare n_dumps_combined vs total_trip_dpr (official DPR cycle count)
if 'total_trip_dpr' in cycle_df.columns:
    valid = cycle_df[(cycle_df.total_trip_dpr>0)&(cycle_df.n_dumps_combined>0)]
    corr = valid[['n_dumps_combined','total_trip_dpr']].corr().iloc[0,1]
    print(f'Correlation between detected cycles and DPR cycles: {corr:.3f}')
    cycle_df['dpr_vs_detected_diff'] = cycle_df['n_dumps_combined'] - cycle_df['total_trip_dpr']

# Summary by mine+shift (methodology validation)
cycle_summary = cycle_df.groupby(['mine','shift']).agg(
    avg_cycles_per_shift = ('n_dumps_combined', 'mean'),
    avg_fuel_per_cycle_L = ('fuel_per_cycle_L', 'mean'),
    avg_runhrs_per_shift = ('runhrs', 'mean'),
    ai_detection_count   = ('n_dumps_ai', 'sum'),
    spatial_detect_count = ('n_dumps_spatial', 'sum'),
    n_shifts_observed    = ('n_dumps_combined', 'count'),
).round(2).reset_index()

print('\nCycle segmentation summary by mine+shift:')
print(cycle_summary.to_string())

# Per-vehicle cycle stats
veh_cycle = cycle_df[cycle_df.n_dumps_combined>0].groupby('vehicle').agg(
    avg_cycles=('n_dumps_combined','mean'),
    avg_fuel_per_cycle_L=('fuel_per_cycle_L','mean'),
    detection_method=('mine', lambda x: 'analog_input' if x.iloc[0]=='mine001' else 'spatial')
).round(2).reset_index()

cycle_summary.to_csv(f'{WORK}/output3_cycle_segmentation_by_route.csv', index=False)
veh_cycle.to_csv(f'{WORK}/output3_cycle_segmentation_by_vehicle.csv', index=False)
cycle_df[['vehicle','date','shift','mine',
           'n_dumps_ai','n_dumps_spatial','n_dumps_combined',
           'acons','runhrs','fuel_per_cycle_L']].to_csv(
    f'{WORK}/output3_cycle_segmentation_detail.csv', index=False
)
print(f'\nSaved: output3_cycle_segmentation_by_route.csv')
print(f'Saved: output3_cycle_segmentation_by_vehicle.csv')
print(f'Saved: output3_cycle_segmentation_detail.csv')

# ══════════════════════════════════════════════════════════════
# OUTPUT 4: DAILY FUEL CONSISTENCY
# Definition: Show that sum(predicted_shift_acons per day) ≈
#             actual daily fuel consumption from DPR.
# Method:
#   For training data: verify sum of 3 shifts = daily total
#   For test predictions: aggregate shift predictions to daily
#   and compare against training-period daily actuals where available
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('OUTPUT 4: Daily Fuel Consistency')
print('='*60)

# ── 4a: Training consistency check ──────────────────────────
train_daily = train_raw.groupby(['vehicle','date'])['acons'].sum().reset_index()
train_daily.columns = ['vehicle','date','daily_acons_actual']

# Rolling prediction for each shift (using same rolling-7 EWM logic as main model)
def build_shift_preds(df):
    df = df.sort_values('date').copy()
    df['pred_ewm7'] = df['acons'].shift(1).ewm(span=7, min_periods=1).mean()
    df['pred_roll7']= df['acons'].shift(1).rolling(7, min_periods=1).mean()
    return df

shift_preds = []
for (veh, shift), grp in train_raw.groupby(['vehicle','shift']):
    shift_preds.append(build_shift_preds(grp))
shift_pred_df = pd.concat(shift_preds, ignore_index=True)

# Sum shift predictions to get daily predicted
daily_pred = shift_pred_df.groupby(['vehicle','date']).agg(
    daily_pred_ewm7 =('pred_ewm7', 'sum'),
    daily_pred_roll7=('pred_roll7','sum'),
).reset_index()

consistency_train = train_daily.merge(daily_pred, on=['vehicle','date'], how='left')
consistency_train['residual_ewm7']  = consistency_train['daily_acons_actual'] - consistency_train['daily_pred_ewm7']
consistency_train['residual_roll7'] = consistency_train['daily_acons_actual'] - consistency_train['daily_pred_roll7']

rmse_ewm  = np.sqrt((consistency_train['residual_ewm7']**2).mean())
rmse_roll = np.sqrt((consistency_train['residual_roll7']**2).mean())
mae_ewm   = consistency_train['residual_ewm7'].abs().mean()
mae_roll  = consistency_train['residual_roll7'].abs().mean()

print(f'Training daily consistency (rolling-7 EWM):  RMSE={rmse_ewm:.2f}L  MAE={mae_ewm:.2f}L')
print(f'Training daily consistency (rolling-7 mean): RMSE={rmse_roll:.2f}L  MAE={mae_roll:.2f}L')

# Per-vehicle consistency
veh_consistency = consistency_train.groupby('vehicle').apply(
    lambda x: pd.Series({
        'daily_rmse_ewm7': np.sqrt((x['residual_ewm7']**2).mean()),
        'daily_mae_ewm7':  x['residual_ewm7'].abs().mean(),
        'mean_actual':     x['daily_acons_actual'].mean(),
        'mean_predicted':  x['daily_pred_ewm7'].mean(),
        'bias':            x['residual_ewm7'].mean(),
        'n_days':          len(x),
    })
).reset_index()

print('\nPer-vehicle daily consistency:')
print(veh_consistency.sort_values('daily_rmse_ewm7', ascending=False).round(2).to_string())

# ── 4b: Test prediction daily consistency ──────────────────
if has_main_preds:
    test_with_preds = id_map.merge(main_sub, on='id', how='left')

    # Aggregate shift-level predictions to daily
    test_daily = test_with_preds.groupby(['vehicle','date'])['predicted_fuel_value'].sum().reset_index()
    test_daily.columns = ['vehicle','date','daily_predicted_L']

    # Physics-based expected daily fuel (sanity check)
    test_daily = test_daily.merge(mine_veh, on='vehicle', how='left')
    test_daily = test_daily.merge(
        route_bm[route_bm.route_shift=='A'][['route_mine','route_fuel_benchmark_L']].rename(
            columns={'route_mine':'mine','route_fuel_benchmark_L':'route_daily_bm_A'}
        ), on='mine', how='left'
    )

    print(f'\nTest daily prediction stats:')
    print(test_daily['daily_predicted_L'].describe())

    # Flag anomalous days: predicted > 3x historical daily average
    hist_daily = train_daily.groupby('vehicle')['daily_acons_actual'].agg(
        hist_daily_mean='mean', hist_daily_std='std', hist_daily_med='median'
    ).reset_index()
    test_daily = test_daily.merge(hist_daily, on='vehicle', how='left')
    test_daily['is_anomaly'] = (
        test_daily['daily_predicted_L'] > test_daily['hist_daily_mean'] + 3*test_daily['hist_daily_std']
    )

    n_anomaly = test_daily['is_anomaly'].sum()
    print(f'\nPotential anomaly days (predicted > mean + 3σ): {n_anomaly}')
    if n_anomaly > 0:
        print(test_daily[test_daily['is_anomaly']][
            ['vehicle','date','daily_predicted_L','hist_daily_mean','hist_daily_std']
        ].to_string())

    test_daily.to_csv(f'{WORK}/output4_daily_consistency_test.csv', index=False)
    print(f'\nSaved: output4_daily_consistency_test.csv')

# ── 4c: Summary consistency table (train + test) ─────────────
consistency_summary = consistency_train.groupby('vehicle').agg(
    train_daily_mean=('daily_acons_actual', 'mean'),
    train_daily_std =('daily_acons_actual', 'std'),
    pred_daily_mean =('daily_pred_ewm7',    'mean'),
    daily_rmse      =('residual_ewm7',      lambda x: np.sqrt((x**2).mean())),
    daily_bias      =('residual_ewm7',      'mean'),
    n_train_days    =('vehicle',            'count'),
).round(2).reset_index()
consistency_summary['pred_accuracy_pct'] = (
    100 * (1 - consistency_summary['daily_rmse'] / consistency_summary['train_daily_mean'].clip(lower=1))
).round(1)

consistency_summary.to_csv(f'{WORK}/output4_daily_consistency_train.csv', index=False)
print(f'\nSaved: output4_daily_consistency_train.csv')
print('\nTraining daily consistency summary:')
print(consistency_summary[['vehicle','train_daily_mean','pred_daily_mean',
                             'daily_rmse','daily_bias','pred_accuracy_pct']].to_string())

# ══════════════════════════════════════════════════════════════
# COMBINED SUMMARY — all 4 outputs in one report table
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('COMBINED REPORT TABLE')
print('='*60)

report = veh_summary[['vehicle','mine','mean_lph_eff_ratio','mean_fuel_excess_L',
                        'overall_efficiency_label']].copy()
report = report.merge(veh_cycle[['vehicle','avg_cycles','avg_fuel_per_cycle_L']], on='vehicle', how='left')
report = report.merge(
    consistency_summary[['vehicle','train_daily_mean','daily_rmse','pred_accuracy_pct']],
    on='vehicle', how='left'
)
# Add physics baseline prediction
phys = id_map.merge(vs_lph, on=['vehicle','shift'], how='left')
phys = phys.merge(rh_stats, on=['vehicle','shift'], how='left')
phys['phys_shift_pred'] = phys['lph_vs_med'] * phys['rh_med']
phys_daily = phys.groupby(['vehicle','date'])['phys_shift_pred'].sum().reset_index()
phys_daily_avg = phys_daily.groupby('vehicle')['phys_shift_pred'].mean().reset_index()
phys_daily_avg.columns = ['vehicle','test_physics_pred_daily']
report = report.merge(phys_daily_avg, on='vehicle', how='left')

report = report.sort_values('mean_fuel_excess_L', ascending=False)
report.to_csv(f'{WORK}/output_combined_report.csv', index=False)
print(report.to_string())
print(f'\nSaved: output_combined_report.csv')

print('\n' + '='*60)
print('ALL SECONDARY OUTPUTS COMPLETE')
print('='*60)
print(f'Files saved to {WORK}:')
print('  output1_route_fuel_benchmark.csv')
print('  output2_dumper_efficiency_by_shift.csv')
print('  output2_dumper_efficiency_summary.csv')
print('  output3_cycle_segmentation_by_route.csv')
print('  output3_cycle_segmentation_by_vehicle.csv')
print('  output3_cycle_segmentation_detail.csv')
print('  output4_daily_consistency_train.csv')
print('  output4_daily_consistency_test.csv  (if main submission exists)')
print('  output_combined_report.csv')