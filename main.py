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
                print("⚠ Local Earth Engine initialization failed. Running fallback mode.")
    except Exception as e:
        print(f"⚠ Earth Engine setup note: {e}")

    if ee_initialized:
        try:
            ROUBAIX_GEOM = ee.Geometry.Point([3.1746, 50.6901])
        except Exception as e:
            print(f"⚠ Could not instantiate geometry: {e}")
            ROUBAIX_GEOM = None

def get_rainfall_ee(dataset_id, start_date, end_date, variable_name):
    """Fetches daily precipitation from Earth Engine or uses stochastic fallback."""
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

    # Fallback rainfall generator representing Roubaix's temperate climate
    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame({'date': dates, 'precip_mm': np.random.gamma(0.6, 3.2, len(dates))})
    df['precip_mm'] = df['precip_mm'].fillna(0)
    return df

def run_water_balance_simulation(roof_area, occupants, precip_series, tank_capacities_m3=[5, 10, 15, 20, 25, 30], tariff_per_m3=2.3961, is_courtyard=False, courtyard_area=0.0):
    # Parameters from Excel sheet 05_HARVESTING_PARAMETERS & 04_WATER_DEMAND
    runoff_c, coll_eff, first_flush, filter_eff = 0.85, 0.90, 0.05, 0.95
    net_harvest_efficiency = runoff_c * coll_eff * (1 - first_flush) * filter_eff

    if not is_courtyard:
        toilet_d  = occupants * 30.0    # WD01: 30 L/person/day
        laundry_d = occupants * 18.0    # WD04: 18 L/person/day
        clean_d   = occupants * 10.0    # WD02: 10 L/person/day (Cleaning + Outdoor)
        irrigation_d = 0.0
        total_daily_demand = toilet_d + laundry_d + clean_d
    else:
        toilet_d, laundry_d, clean_d = 0.0, 0.0, 0.0
        # Courtyard irrigation demand based on sheet 03_COURTYARD & 07_SCEN1_SEASONAL_IRRIGATION (~3.5 L/m2/day peak equivalent)
        irrigation_d = courtyard_area * 3.5
        total_daily_demand = irrigation_d

    capacity_performance = {}

    for cap_m3 in tank_capacities_m3:
        tank_cap_L = cap_m3 * 1000.0
        storage_L = tank_cap_L * 0.5  # Initial 50% tank level
        
        storage_time_series = []
        supplied_total, unmet_total, overflow_total = 0.0, 0.0, 0.0
        potential_harvest_total, collected_total = 0.0, 0.0
        empty_days = 0

        for precip_mm in precip_series:
            area = courtyard_area if is_courtyard else roof_area
            potential_harvest = area * precip_mm
            collected = potential_harvest * net_harvest_efficiency
            
            potential_harvest_total += potential_harvest
            collected_total += collected
            storage_L += collected
            
            if storage_L > tank_cap_L:
                overflow_total += (storage_L - tank_cap_L)
                storage_L = tank_cap_L
                
            if storage_L >= total_daily_demand:
                supplied_total += total_daily_demand
                storage_L -= total_daily_demand
            else:
                supplied_total += storage_L
                unmet_total += (total_daily_demand - storage_L)
                storage_L = 0.0
                empty_days += 1
                
            storage_time_series.append(round(storage_L, 1))

        total_demand_period = total_daily_demand * len(precip_series)
        coverage_pct = (supplied_total / total_demand_period * 100) if total_demand_period > 0 else 0
        capture_rate = (collected_total / potential_harvest_total * 100) if potential_harvest_total > 0 else 0
        reuse_rate = (supplied_total / collected_total * 100) if collected_total > 0 else 0
        runoff_reduction_pct = (supplied_total / (potential_harvest_total * runoff_c) * 100) if potential_harvest_total > 0 else 0
        
        annual_cost_savings = (supplied_total / 1000.0) * tariff_per_m3
        savings_toilet = (supplied_total * (toilet_d / total_daily_demand if total_daily_demand > 0 else 0) / 1000.0) * tariff_per_m3
        savings_laundry = (supplied_total * (laundry_d / total_daily_demand if total_daily_demand > 0 else 0) / 1000.0) * tariff_per_m3
        savings_cleaning = (supplied_total * (clean_d / total_daily_demand if total_daily_demand > 0 else 0) / 1000.0) * tariff_per_m3
        savings_irrigation = (supplied_total * (irrigation_d / total_daily_demand if total_daily_demand > 0 else 0) / 1000.0) * tariff_per_m3

        capacity_performance[f"{cap_m3}m3"] = {
            "Potential_Harvested_L": round(potential_harvest_total, 1),
            "Collected_Rainwater_L": round(collected_total, 1),
            "Toilet_Demand_L": round(toilet_d * len(precip_series), 1),
            "Laundry_Demand_L": round(laundry_d * len(precip_series), 1),
            "Cleaning_Demand_L": round(clean_d * len(precip_series), 1),
            "Irrigation_Demand_L": round(irrigation_d * len(precip_series), 1),
            "Total_Water_Demand_L": round(total_demand_period, 1),
            "Water_Supplied_L": round(supplied_total, 1),
            "Water_Coverage_Pct": round(coverage_pct, 2),
            "Capture_Rate_Pct": round(capture_rate, 2),
            "Reuse_Rate_Pct": round(reuse_rate, 2),
            "Runoff_Reduction_Pct": round(runoff_reduction_pct, 2),
            "Unmet_Water_Demand_L": round(unmet_total, 1),
            "Overflow_Volume_L": round(overflow_total, 1),
            "Empty_Days": empty_days,
            "Cost_Savings_Total_EUR": round(annual_cost_savings, 2),
            "Savings_Toilet_EUR": round(savings_toilet, 2),
            "Savings_Laundry_EUR": round(savings_laundry, 2),
            "Savings_Cleaning_EUR": round(savings_cleaning, 2),
            "Savings_Irrigation_EUR": round(savings_irrigation, 2),
            "Storage_Level_Over_Time_L": storage_time_series
        }

    return capacity_performance

def main():
    print("Starting Courée Blasin Advanced Water Balance Pipeline...")
    init_earth_engine()
    
    bldgs = pd.read_excel(EXCEL_FILE, sheet_name='01_BUILDINGS')
    roofs = pd.read_excel(EXCEL_FILE, sheet_name='02_ROOFS')
    yard  = pd.read_excel(EXCEL_FILE, sheet_name='03_COURTYARD')
    tariff_df = pd.read_excel(EXCEL_FILE, sheet_name='09_WATER_TARIFFS')
    
    pv_tariff = tariff_df['PV (€/m³)'].values[0] + \
                tariff_df['Redevance lutte contre la pollution (€/m³)'].values[0] + \
                tariff_df['Redevance modernisation des réseaux (€/m³)'].values[0] + \
                tariff_df['Redevance prélèvement ressource en eau (€/m³)'].values[0] + \
                tariff_df['Redevance Voies navigables de France (€/m³)'].values[0]

    df_merged = pd.merge(bldgs, roofs[['Building_ID', 'Collectable_Roof_Area_m²']], on='Building_ID')

    # Rainfall time series across periods: 2014, 2026, 2030, This Week, Next Week
    rain_2014 = get_rainfall_ee('NASA/GDDP-CMIP6', '2014-01-01', '2014-12-31', 'pr')['precip_mm'].values
    rain_2026 = get_rainfall_ee('NASA/GDDP-CMIP6', '2026-01-01', '2026-12-31', 'pr')['precip_mm'].values
    rain_2030 = get_rainfall_ee('NASA/GDDP-CMIP6', '2030-01-01', '2030-12-31', 'pr')['precip_mm'].values
    rain_this_week = get_rainfall_ee('projects/gcp-public-data-weathernext/assets/weathernext_2_0_0_mean', '2026-09-03', '2026-09-09', 'total_precipitation')['precip_mm'].values
    rain_next_week = get_rainfall_ee('projects/gcp-public-data-weathernext/assets/weathernext_2_0_0_mean', '2026-09-10', '2026-09-16', 'total_precipitation')['precip_mm'].values

    output_database = {}

    # Process each building separately
    for _, row in df_merged.iterrows():
        b_id = row['Building_ID']
        occ  = row['Occupants']
        roof_area = row['Collectable_Roof_Area_m²']

        output_database[b_id] = {
            "QGIS_Building_I": b_id,
            "Occupants": occ,
            "Roof_Area_m2": roof_area,
            "Results_2014": run_water_balance_simulation(roof_area, occ, rain_2014, tariff_per_m3=pv_tariff),
            "Results_2026": run_water_balance_simulation(roof_area, occ, rain_2026, tariff_per_m3=pv_tariff),
            "Results_2030": run_water_balance_simulation(roof_area, occ, rain_2030, tariff_per_m3=pv_tariff),
            "Results_This_Week": run_water_balance_simulation(roof_area, occ, rain_this_week, tariff_per_m3=pv_tariff),
            "Results_Next_Week_Forecast": run_water_balance_simulation(roof_area, occ, rain_next_week, tariff_per_m3=pv_tariff)
        }

    # Process courtyard separately
    courtyard_area = yard['Courtyard_Area'].values[0]
    output_database['CY001'] = {
        "QGIS_CY_ID": "CY001",
        "Courtyard_Area_m2": courtyard_area,
        "Results_2014": run_water_balance_simulation(0, 0, rain_2014, tariff_per_m3=pv_tariff, is_courtyard=True, courtyard_area=courtyard_area),
        "Results_2026": run_water_balance_simulation(0, 0, rain_2026, tariff_per_m3=pv_tariff, is_courtyard=True, courtyard_area=courtyard_area),
        "Results_2030": run_water_balance_simulation(0, 0, rain_2030, tariff_per_m3=pv_tariff, is_courtyard=True, courtyard_area=courtyard_area),
        "Results_This_Week": run_water_balance_simulation(0, 0, rain_this_week, tariff_per_m3=pv_tariff, is_courtyard=True, courtyard_area=courtyard_area),
        "Results_Next_Week_Forecast": run_water_balance_simulation(0, 0, rain_next_week, tariff_per_m3=pv_tariff, is_courtyard=True, courtyard_area=courtyard_area)
    }
    
    qgis_flat_output = []
    
    for b_id, data in output_database.items():
        cap_key = "20m3"
        if b_id == 'CY001':
            flat_entry = {
                "Building_I": b_id,
                "Occupants": 0,
                "Roof_m2": data["Courtyard_Area_m2"],
                "Cov_2014_Pct": data["Results_2014"][cap_key]["Water_Coverage_Pct"],
                "Unmet_2014_L": data["Results_2014"][cap_key]["Unmet_Water_Demand_L"],
                "Overflow_2014_L": data["Results_2014"][cap_key]["Overflow_Volume_L"],
                "Savings_2014_EUR": data["Results_2014"][cap_key]["Cost_Savings_Total_EUR"],
                "Cov_2026_Pct": data["Results_2026"][cap_key]["Water_Coverage_Pct"],
                "Unmet_2026_L": data["Results_2026"][cap_key]["Unmet_Water_Demand_L"],
                "Overflow_2026_L": data["Results_2026"][cap_key]["Overflow_Volume_L"],
                "Savings_2026_EUR": data["Results_2026"][cap_key]["Cost_Savings_Total_EUR"],
                "Cov_2030_Pct": data["Results_2030"][cap_key]["Water_Coverage_Pct"],
                "Supply_ThisWeek_L": data["Results_This_Week"][cap_key]["Water_Supplied_L"],
                "Supply_NextWeek_L": data["Results_Next_Week_Forecast"][cap_key]["Water_Supplied_L"]
            }
        else:
            flat_entry = {
                "Building_I": b_id,
                "Occupants": data["Occupants"],
                "Roof_m2": data["Roof_Area_m2"],
                "Cov_2014_Pct": data["Results_2014"][cap_key]["Water_Coverage_Pct"],
                "Unmet_2014_L": data["Results_2014"][cap_key]["Unmet_Water_Demand_L"],
                "Overflow_2014_L": data["Results_2014"][cap_key]["Overflow_Volume_L"],
                "Savings_2014_EUR": data["Results_2014"][cap_key]["Cost_Savings_Total_EUR"],
                "Cov_2026_Pct": data["Results_2026"][cap_key]["Water_Coverage_Pct"],
                "Unmet_2026_L": data["Results_2026"][cap_key]["Unmet_Water_Demand_L"],
                "Overflow_2026_L": data["Results_2026"][cap_key]["Overflow_Volume_L"],
                "Savings_2026_EUR": data["Results_2026"][cap_key]["Cost_Savings_Total_EUR"],
                "Cov_2030_Pct": data["Results_2030"][cap_key]["Water_Coverage_Pct"],
                "Supply_ThisWeek_L": data["Results_This_Week"][cap_key]["Water_Supplied_L"],
                "Supply_NextWeek_L": data["Results_Next_Week_Forecast"][cap_key]["Water_Supplied_L"]
            }
        qgis_flat_output.append(flat_entry)

    with open("couree_water_results.json", "w") as f:
        json.dump(output_database, f, indent=4)
        
    with open("couree_qgis_flat.json", "w") as f:
        json.dump(qgis_flat_output, f, indent=4)
        
    df_qgis_csv = pd.DataFrame(qgis_flat_output)
    df_qgis_csv.to_csv("couree_qgis_flat.csv", index=False)

    print("✅ Advanced pipeline execution successful. All calculations updated.")

if __name__ == "__main__":
    main()
