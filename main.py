import os
import json
import pandas as pd
import numpy as np
import ee

EXCEL_FILE = "Couree_Blasin_Roof_Assumption_Prototype (2).xlsx"

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
    """Fetches daily precipitation from Earth Engine or uses realistic fallback."""
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

def run_comprehensive_simulation(roof_area, occupants, precip_series, tank_capacity_m3=10, tariff_per_m3=2.3961, is_courtyard=False, courtyard_area=0.0, monthly_irrigation_depths=None):
    runoff_c, coll_eff, first_flush, filter_eff = 0.85, 0.90, 0.05, 0.95
    net_harvest_efficiency = runoff_c * coll_eff * (1 - first_flush) * filter_eff

    tank_cap_L = tank_capacity_m3 * 1000.0
    storage_L = tank_cap_L * 0.5
    
    potential_harvest_total, collected_total = 0.0, 0.0
    demand_toilet, demand_cleaning, demand_laundry, demand_irrigation = 0.0, 0.0, 0.0, 0.0
    supplied_total, unmet_total, overflow_total = 0.0, 0.0, 0.0
    empty_days = 0
    storage_levels = []

    dates = pd.to_datetime([p['date'] if isinstance(p, dict) and 'date' in p else pd.Timestamp('2026-01-01') + pd.Timedelta(days=i) for i, p in enumerate(precip_series)])
    precip_values = [p['precip_mm'] if isinstance(p, dict) else p for p in precip_series]

    for i, precip_mm in enumerate(precip_values):
        current_date = dates[i] if i < len(dates) else pd.Timestamp('2026-01-01')
        month_name = current_date.strftime('%B')

        if not is_courtyard:
            dt = occupants * 30.0   # Toilet
            dc = occupants * 10.0   # Cleaning + Outdoor
            dl = occupants * 18.0   # Laundry
            daily_demand = dt + dc + dl
            demand_toilet += dt
            demand_cleaning += dc
            demand_laundry += dl
        else:
            weekly_depth = monthly_irrigation_depths.get(month_name, 0.0) if monthly_irrigation_depths else 0.0
            daily_demand = courtyard_area * (weekly_depth / 7.0)
            demand_irrigation += daily_demand

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
            
        storage_levels.append(round(storage_L, 1))

    total_demand = demand_toilet + demand_cleaning + demand_laundry + demand_irrigation
    if total_demand == 0:
        total_demand = 1.0

    return {
        "Potential_Harvest_L": round(potential_harvest_total, 1),
        "Collected_Rainwater_L": round(collected_total, 1),
        "Total_Demand_L": round(total_demand, 1),
        "Toilet_Demand_L": round(demand_toilet, 1),
        "Cleaning_Demand_L": round(demand_cleaning, 1),
        "Laundry_Demand_L": round(demand_laundry, 1),
        "Irrigation_Demand_L": round(demand_irrigation, 1),
        "Water_Supplied_L": round(supplied_total, 1),
        "Water_Coverage_Pct": round((supplied_total / total_demand) * 100, 2),
        "Capture_Rate_Pct": round((collected_total / potential_harvest_total) * 100, 2) if potential_harvest_total > 0 else 0,
        "Reuse_Rate_Pct": round((supplied_total / collected_total) * 100, 2) if collected_total > 0 else 0,
        "Runoff_Reduction_Pct": round((supplied_total / potential_harvest_total) * 100, 2) if potential_harvest_total > 0 else 0,
        "Unmet_Demand_L": round(unmet_total, 1),
        "Overflow_Volume_L": round(overflow_total, 1),
        "Empty_Days": empty_days,
        "Cost_Savings_EUR": round((supplied_total / 1000.0) * tariff_per_m3, 2),
        "Avg_Storage_L": round(np.mean(storage_levels), 1)
    }

def main():
    print("Starting Multi-Dataset Courée Blasin Pipeline...")
    init_earth_engine()
    
    bldgs = pd.read_excel(EXCEL_FILE, sheet_name='01_BUILDINGS')
    roofs = pd.read_excel(EXCEL_FILE, sheet_name='02_ROOFS')
    yard  = pd.read_excel(EXCEL_FILE, sheet_name='03_COURTYARD')
    irr_df = pd.read_excel(EXCEL_FILE, sheet_name='07_SCEN1_SEASONAL_IRRIGATION')
    tariff_df = pd.read_excel(EXCEL_FILE, sheet_name='09_WATER_TARIFFS')
    gee_df = pd.read_excel(EXCEL_FILE, sheet_name='10_GEE_DATASETS')
    
    monthly_irrigation_depths = dict(zip(irr_df['Month'], irr_df['Water_Depth_mm_per_week']))
    pv_tariff = tariff_df['PV (€/m³)'].values[0] + tariff_df['Redevance lutte contre la pollution (€/m³)'].values[0] + tariff_df['Redevance modernisation des réseaux (€/m³)'].values[0] + tariff_df['Redevance prélèvement ressource en eau (€/m³)'].values[0] + tariff_df['Redevance Voies navigables de France (€/m³)'].values[0]

    df_merged = pd.merge(bldgs, roofs[['Building_ID', 'Collectable_Roof_Area_m²']], on='Building_ID')
    dataset_id = gee_df['Dataset_ID'].values[0] if 'Dataset_ID' in gee_df.columns else 'NASA/GDDP-CMIP6'
    
    rain_2014 = get_rainfall_ee(dataset_id, '2014-01-01', '2014-12-31', 'pr')
    rain_2026 = get_rainfall_ee(dataset_id, '2026-01-01', '2026-12-31', 'pr')
    rain_2030 = get_rainfall_ee(dataset_id, '2030-01-01', '2030-12-31', 'pr')
    rain_this_wk = get_rainfall_ee(dataset_id, '2026-09-13', '2026-09-19', 'pr')
    rain_next_wk = get_rainfall_ee(dataset_id, '2026-09-20', '2026-09-26', 'pr')

    # --- DATASET 1: INDIVIDUAL BUILDINGS BASELINE ---
    baseline_records = []
    indiv_coverages = []
    for _, row in df_merged.iterrows():
        b_id = row['Building_ID']
        occ = row['Occupants']
        area = row['Collectable_Roof_Area_m²']

        s2014 = run_comprehensive_simulation(area, occ, rain_2014['precip_mm'].values, 10, pv_tariff)
        s2026 = run_comprehensive_simulation(area, occ, rain_2026['precip_mm'].values, 10, pv_tariff)
        s2030 = run_comprehensive_simulation(area, occ, rain_2030['precip_mm'].values, 10, pv_tariff)
        sthis = run_comprehensive_simulation(area, occ, rain_this_wk['precip_mm'].values, 10, pv_tariff)
        snext = run_comprehensive_simulation(area, occ, rain_next_wk['precip_mm'].values, 10, pv_tariff)

        indiv_coverages.append(s2026["Water_Coverage_Pct"])

        baseline_records.append({
            "Building_ID": b_id,
            "Occupants": occ,
            "Roof_Area_m2": area,
            # 2026 Yearly
            "Pot_Harvest_2026_L": s2026["Potential_Harvest_L"],
            "Collected_2026_L": s2026["Collected_Rainwater_L"],
            "Total_Demand_2026_L": s2026["Total_Demand_L"],
            "Toilet_Demand_2026_L": s2026["Toilet_Demand_L"],
            "Cleaning_Demand_2026_L": s2026["Cleaning_Demand_L"],
            "Laundry_Demand_2026_L": s2026["Laundry_Demand_L"],
            "Supplied_2026_L": s2026["Water_Supplied_L"],
            "Coverage_2026_Pct": s2026["Water_Coverage_Pct"],
            "Capture_Rate_2026_Pct": s2026["Capture_Rate_Pct"],
            "Reuse_Rate_2026_Pct": s2026["Reuse_Rate_Pct"],
            "Runoff_Reduction_2026_Pct": s2026["Runoff_Reduction_Pct"],
            "Savings_2026_EUR": s2026["Cost_Savings_EUR"],
            "Unmet_Demand_2026_L": s2026["Unmet_Demand_L"],
            "Overflow_2026_L": s2026["Overflow_Volume_L"],
            "Empty_Days_2026": s2026["Empty_Days"],
            # Temporal
            "Coverage_2014_Pct": s2014["Water_Coverage_Pct"],
            "Savings_2014_EUR": s2014["Cost_Savings_EUR"],
            "Coverage_2030_Pct": s2030["Water_Coverage_Pct"],
            "Savings_2030_EUR": s2030["Cost_Savings_EUR"],
            # Weekly
            "Coverage_ThisWeek_Pct": sthis["Water_Coverage_Pct"],
            "Coverage_NextWeek_Pct": snext["Water_Coverage_Pct"],
            # Tank capacities
            "Cap_5m3_Cov": run_comprehensive_simulation(area, occ, rain_2026['precip_mm'].values, 5, pv_tariff)["Water_Coverage_Pct"],
            "Cap_20m3_Cov": run_comprehensive_simulation(area, occ, rain_2026['precip_mm'].values, 20, pv_tariff)["Water_Coverage_Pct"]
        })
    pd.DataFrame(baseline_records).to_csv("couree_baseline_buildings.csv", index=False)

    # --- DATASET 2: SCENARIO 1 (COURTYARD IRRIGATION) ---
    cy_area = yard['Courtyard_Area'].values[0]
    cy2014 = run_comprehensive_simulation(0, 0, rain_2014['precip_mm'].values, 10, pv_tariff, True, cy_area, monthly_irrigation_depths)
    cy2026 = run_comprehensive_simulation(0, 0, rain_2026['precip_mm'].values, 10, pv_tariff, True, cy_area, monthly_irrigation_depths)
    cy2030 = run_comprehensive_simulation(0, 0, rain_2030['precip_mm'].values, 10, pv_tariff, True, cy_area, monthly_irrigation_depths)
    
    scenario1_records = [{
        "Courtyard_ID": yard['Courtyard_ID'].values[0],
        "Courtyard_Name": yard['Courtyard_Name'].values[0],
        "Area_m2": cy_area,
        "Pot_Harvest_2026_L": cy2026["Potential_Harvest_L"],
        "Collected_2026_L": cy2026["Collected_Rainwater_L"],
        "Irrigation_Demand_2026_L": cy2026["Irrigation_Demand_L"],
        "Supplied_2026_L": cy2026["Water_Supplied_L"],
        "Coverage_2026_Pct": cy2026["Water_Coverage_Pct"],
        "Savings_2026_EUR": cy2026["Cost_Savings_EUR"],
        "Coverage_2014_Pct": cy2014["Water_Coverage_Pct"],
        "Coverage_2030_Pct": cy2030["Water_Coverage_Pct"]
    }]
    pd.DataFrame(scenario1_records).to_csv("couree_scenario1_courtyard.csv", index=False)

    # --- DATASET 3: SCENARIO 2 (SHARED TANKS & EQUITY INDEX) ---
    total_roof = df_merged['Collectable_Roof_Area_m²'].sum()
    total_occ = df_merged['Occupants'].sum()
    shared_20m3 = run_comprehensive_simulation(total_roof, total_occ, rain_2026['precip_mm'].values, 20, pv_tariff)
    
    mean_cov = np.mean(indiv_coverages)
    std_cov = np.std(indiv_coverages)
    equity_idx = round(1.0 - (std_cov / mean_cov if mean_cov > 0 else 0), 3)

    scenario2_records = [{
        "Configuration": "Single Shared 20m3 Tank",
        "Total_Roof_m2": total_roof,
        "Total_Occupants": total_occ,
        "Shared_Coverage_Pct": shared_20m3["Water_Coverage_Pct"],
        "Shared_Savings_EUR": shared_20m3["Cost_Savings_EUR"],
        "Equity_Index": equity_idx
    }]
    pd.DataFrame(scenario2_records).to_csv("couree_scenario2_shared.csv", index=False)

    print("✅ Successfully generated separate CSV files for Baseline, Scenario 1, and Scenario 2!")

if __name__ == "__main__":
    main()
