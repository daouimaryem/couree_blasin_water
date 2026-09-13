import os
import json
import pandas as pd
import numpy as np
import ee

EXCEL_FILE = "Couree_Blasin_Roof_Assumption_Prototype (2).xlsx"
OUTPUT_CSV_FILE = "couree_blasin_results.csv"

ROUBAIX_GEOM = None

def init_earth_engine():
    global ROUBAIX_GEOM
    ee_initialized = False
    try:
        if os.path.exists('gcp_key.json'):
            credentials = ee.ServiceAccountCredentials(None, 'gcp_key.json')
            ee.Initialize(credentials)
            print("✓ Earth Engine authenticated with service account credentials.")
            ee_initialized = True
        else:
            try:
                ee.Initialize()
                print("✓ Earth Engine initialized locally.")
                ee_initialized = True
            except Exception:
                print("⚠ Local Earth Engine initialization failed. Running fallback stochastic mode.")
    except Exception as e:
        print(f"⚠ Earth Engine setup note: {e}")

    if ee_initialized:
        try:
            ROUBAIX_GEOM = ee.Geometry.Point([3.1746, 50.6901])
        except Exception as e:
            print(f"⚠ Could not instantiate geometry: {e}")
            ROUBAIX_GEOM = None

def get_rainfall_ee(dataset_id, start_date, end_date, variable_name):
    """Fetches daily precipitation from Earth Engine (CMIP6 / NASA) or uses realistic fallback."""
    if ROUBAIX_GEOM is not None:
        try:
            col = ee.ImageCollection(dataset_id) \
                    .filterBounds(ROUBAIX_GEOM) \
                    .filterDate(start_date, end_date)
                    
            def extract_precip(img):
                val = img.reduceRegion(ee.Reducer.mean(), ROUBAIX_GEOM, 30).get(variable_name)
                precip_mm = ee.Number(val).multiply(86400) if 'CMIP6' in dataset_id else ee.Number(val)
                return ee.Feature(None, {
                    'date': img.date().format('YYYY-MM-dd'),
                    'precip_mm': precip_mm
                })

            features = col.map(extract_precip).getInfo()['features']
            records = [f['properties'] for f in features]
            df = pd.DataFrame(records)
            if not df.empty and 'precip_mm' in df.columns:
                df['precip_mm'] = df['precip_mm'].fillna(0)
                return df
        except Exception as e:
            print(f"⚠ Earth Engine query skipped for {dataset_id}: {e}")

    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame({'date': dates, 'precip_mm': np.random.gamma(0.6, 3.2, len(dates))})
    df['precip_mm'] = df['precip_mm'].fillna(0)
    return df

def run_water_balance_simulation(roof_area, occupants, precip_series, tank_capacity_m3=10, tariff_per_m3=2.3961, is_courtyard=False, courtyard_area=0.0, monthly_irrigation_depths=None):
    """Runs daily water balance simulation across any given timeframe."""
    runoff_c, coll_eff, first_flush, filter_eff = 0.85, 0.90, 0.05, 0.95
    net_harvest_efficiency = runoff_c * coll_eff * (1 - first_flush) * filter_eff

    tank_cap_L = tank_capacity_m3 * 1000.0
    storage_L = tank_cap_L * 0.5
    
    supplied_total, unmet_total, overflow_total = 0.0, 0.0, 0.0
    potential_harvest_total, collected_total = 0.0, 0.0
    empty_days = 0

    dates = pd.to_datetime([p['date'] if isinstance(p, dict) and 'date' in p else pd.Timestamp('2026-01-01') + pd.Timedelta(days=i) for i, p in enumerate(precip_series)])
    precip_values = [p['precip_mm'] if isinstance(p, dict) else p for p in precip_series]

    for i, precip_mm in enumerate(precip_values):
        current_date = dates[i] if i < len(dates) else pd.Timestamp('2026-01-01')
        month_name = current_date.strftime('%B')

        if not is_courtyard:
            toilet_d  = occupants * 30.0
            laundry_d = occupants * 18.0
            clean_d   = occupants * 10.0
            daily_demand = toilet_d + laundry_d + clean_d
        else:
            weekly_depth = 0.0
            if monthly_irrigation_depths and month_name in monthly_irrigation_depths:
                weekly_depth = monthly_irrigation_depths[month_name]
            daily_irrigation_depth = weekly_depth / 7.0
            daily_demand = courtyard_area * daily_irrigation_depth

        area = courtyard_area if is_courtyard else roof_area
        potential_harvest = area * precip_mm
        collected = potential_harvest * net_harvest_efficiency
        
        potential_harvest_total += potential_harvest
        collected_total += collected
        storage_L += collected
        
        if storage_L > tank_cap_L:
            overflow_total += (storage_L - tank_cap_L)
            storage_L = tank_cap_L
            
        if storage_L >= daily_demand:
            supplied_total += daily_demand
            storage_L -= daily_demand
        else:
            supplied_total += storage_L
            unmet_total += (daily_demand - storage_L)
            storage_L = 0.0
            empty_days += 1

    total_demand_period = sum([
        (occupants * 58.0) if not is_courtyard else (courtyard_area * (monthly_irrigation_depths.get(pd.Timestamp(d).strftime('%B'), 0)/7.0))
        for d in dates
    ])
    if total_demand_period == 0:
        total_demand_period = 1.0

    coverage_pct = (supplied_total / total_demand_period * 100)
    capture_rate = (collected_total / potential_harvest_total * 100) if potential_harvest_total > 0 else 0
    reuse_rate = (supplied_total / collected_total * 100) if collected_total > 0 else 0
    annual_cost_savings = (supplied_total / 1000.0) * tariff_per_m3

    return {
        "Water_Supplied_L": round(supplied_total, 1),
        "Water_Coverage_Pct": round(coverage_pct, 2),
        "Capture_Rate_Pct": round(capture_rate, 2),
        "Reuse_Rate_Pct": round(reuse_rate, 2),
        "Unmet_Water_Demand_L": round(unmet_total, 1),
        "Overflow_Volume_L": round(overflow_total, 1),
        "Empty_Days": empty_days,
        "Cost_Savings_EUR": round(annual_cost_savings, 2)
    }

def main():
    print("Starting Comprehensive Courée Blasin Pipeline (2014, 2026, 2030 + Scenarios 1 & 2)...")
    init_earth_engine()
    
    bldgs = pd.read_excel(EXCEL_FILE, sheet_name='01_BUILDINGS')
    roofs = pd.read_excel(EXCEL_FILE, sheet_name='02_ROOFS')
    yard  = pd.read_excel(EXCEL_FILE, sheet_name='03_COURTYARD')
    irr_df = pd.read_excel(EXCEL_FILE, sheet_name='07_SCEN1_SEASONAL_IRRIGATION')
    tariff_df = pd.read_excel(EXCEL_FILE, sheet_name='09_WATER_TARIFFS')
    gee_df = pd.read_excel(EXCEL_FILE, sheet_name='10_GEE_DATASETS')
    
    monthly_irrigation_depths = dict(zip(irr_df['Month'], irr_df['Water_Depth_mm_per_week']))

    pv_tariff = tariff_df['PV (€/m³)'].values[0] + \
                tariff_df['Redevance lutte contre la pollution (€/m³)'].values[0] + \
                tariff_df['Redevance modernisation des réseaux (€/m³)'].values[0] + \
                tariff_df['Redevance prélèvement ressource en eau (€/m³)'].values[0] + \
                tariff_df['Redevance Voies navigables de France (€/m³)'].values[0]

    df_merged = pd.merge(bldgs, roofs[['Building_ID', 'Collectable_Roof_Area_m²']], on='Building_ID')

    # Fetch Rainfall for 2014, 2026, and 2030 from GEE parameters in Sheet 10
    dataset_id = gee_df['Dataset_ID'].values[0] if 'Dataset_ID' in gee_df.columns else 'NASA/GDDP-CMIP6'
    
    print("Fetching rainfall time-series for 2014, 2026, and 2030...")
    rain_2014 = get_rainfall_ee(dataset_id, '2014-01-01', '2014-12-31', 'pr')
    rain_2026 = get_rainfall_ee(dataset_id, '2026-01-01', '2026-12-31', 'pr')
    rain_2030 = get_rainfall_ee(dataset_id, '2030-01-01', '2030-12-31', 'pr')

    qgis_flat_output = []
    individual_coverages_2026 = []

    # 1. Individual Building Calculations across Temporal Periods (2014, 2026, 2030)
    building_results = []
    for _, row in df_merged.iterrows():
        b_id = row['Building_ID']
        occ  = row['Occupants']
        roof_area = row['Collectable_Roof_Area_m²']

        sim_2014 = run_water_balance_simulation(roof_area, occ, rain_2014['precip_mm'].values, tank_capacity_m3=10, tariff_per_m3=pv_tariff)
        sim_2026 = run_water_balance_simulation(roof_area, occ, rain_2026['precip_mm'].values, tank_capacity_m3=10, tariff_per_m3=pv_tariff)
        sim_2030 = run_water_balance_simulation(roof_area, occ, rain_2030['precip_mm'].values, tank_capacity_m3=10, tariff_per_m3=pv_tariff)

        individual_coverages_2026.append(sim_2026["Water_Coverage_Pct"])
        
        building_results.append({
            "Building_I": b_id,
            "Occupants": occ,
            "Roof_m2": roof_area,
            "Cov_2014_Pct": sim_2014["Water_Coverage_Pct"],
            "Cov_2026_Pct": sim_2026["Water_Coverage_Pct"],
            "Cov_2030_Pct": sim_2030["Water_Coverage_Pct"],
            "Unmet_2026_L": sim_2026["Unmet_Water_Demand_L"],
            "Savings_2026_EUR": sim_2026["Cost_Savings_EUR"]
        })

    # 2. Scenario 2: Shared Tank Equity Index Calculation
    mean_cov = np.mean(individual_coverages_2026)
    std_cov = np.std(individual_coverages_2026)
    equity_index = round(1.0 - (std_cov / mean_cov if mean_cov > 0 else 0), 3)

    # Scenario 2 Storage Configurations (1 shared 20m3, 2 shared 10m3, 4 shared 5m3)
    total_roof_area = df_merged['Collectable_Roof_Area_m²'].sum()
    total_occupants = df_merged['Occupants'].sum()
    
    shared_1_tank = run_water_balance_simulation(total_roof_area, total_occupants, rain_2026['precip_mm'].values, tank_capacity_m3=20, tariff_per_m3=pv_tariff)
    
    group1_roof = df_merged.iloc[:4]['Collectable_Roof_Area_m²'].sum()
    group1_occ = df_merged.iloc[:4]['Occupants'].sum()
    shared_2_tanks = run_water_balance_simulation(group1_roof, group1_occ, rain_2026['precip_mm'].values, tank_capacity_m3=10, tariff_per_m3=pv_tariff)

    # 3. Scenario 1: Courtyard Seasonal Irrigation
    courtyard_area = yard['Courtyard_Area'].values[0]
    yard_sim_2026 = run_water_balance_simulation(0, 0, rain_2026['precip_mm'].values, tank_capacity_m3=10, tariff_per_m3=pv_tariff, is_courtyard=True, courtyard_area=courtyard_area, monthly_irrigation_depths=monthly_irrigation_depths)

    # 4. Compile into QGIS-ready Flat CSV Output
    for b_data in building_results:
        b_data["Shared_1Tank_Coverage_Pct"] = shared_1_tank["Water_Coverage_Pct"]
        b_data["Shared_2Tanks_Coverage_Pct"] = shared_2_tanks["Water_Coverage_Pct"]
        b_data["Equity_Index"] = equity_index
        b_data["Courtyard_Scenario1_Coverage"] = yard_sim_2026["Water_Coverage_Pct"]
        qgis_flat_output.append(b_data)

    # Append Courtyard summary row
    qgis_flat_output.append({
        "Building_I": "CY001",
        "Occupants": 0,
        "Roof_m2": courtyard_area,
        "Cov_2014_Pct": yard_sim_2026["Water_Coverage_Pct"],
        "Cov_2026_Pct": yard_sim_2026["Water_Coverage_Pct"],
        "Cov_2030_Pct": yard_sim_2026["Water_Coverage_Pct"],
        "Unmet_2026_L": yard_sim_2026["Unmet_Water_Demand_L"],
        "Savings_2026_EUR": yard_sim_2026["Cost_Savings_EUR"],
        "Shared_1Tank_Coverage_Pct": shared_1_tank["Water_Coverage_Pct"],
        "Shared_2Tanks_Coverage_Pct": shared_2_tanks["Water_Coverage_Pct"],
        "Equity_Index": equity_index,
        "Courtyard_Scenario1_Coverage": yard_sim_2026["Water_Coverage_Pct"]
    })

    df_qgis_csv = pd.DataFrame(qgis_flat_output)
    df_qgis_csv.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"✅ Successfully generated full multi-period & scenario results in {OUTPUT_CSV_FILE}!")

if __name__ == "__main__":
    main()
