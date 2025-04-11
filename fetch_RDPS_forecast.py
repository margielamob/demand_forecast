# --- Start of Script ---
import os
import requests
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta # Use timezone for UTC clarity
from tqdm import tqdm
import time # For potential sleep

# Ensure necessary libraries are installed:
# pip install requests xarray pandas numpy tqdm cfgrib
# NOTE: Requires eccodes library installation. Check previous ECCODES ERROR messages
# regarding JPEG support. You might need platform admin help for a fix if needed.

# =======================
# CONFIGURATION SETTINGS
# =======================

# --- Base URL for RDPS 10km GRIB2 data ---
model_base_url = "https://dd.weather.gc.ca/model_gem_regional/10km/grib2"

# --- Automatically Determine Latest Available Model Run ---
now_utc = datetime.now(timezone.utc)
print(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
possible_run_hours = [0, 6, 12, 18]
# How long to wait after a run starts before assuming data is available
availability_delay = timedelta(hours=3.5) # +/- 3.5 hours is often safe for RDPS

target_run_hour_int = None
target_run_datetime_obj = None

# Look backwards from the latest possible run time
for h in sorted(possible_run_hours, reverse=True):
    potential_run_start = now_utc.replace(hour=h, minute=0, second=0, microsecond=0)
    if potential_run_start > now_utc: # If run time is in the future, check yesterday's
         potential_run_start -= timedelta(days=1)
    # Check if this run should be available
    if potential_run_start + availability_delay <= now_utc:
        target_run_hour_int = h
        target_run_datetime_obj = potential_run_start
        break # Stop searching once the latest available is found

# Fallback if no run seems available
if target_run_datetime_obj is None:
    print("Warning: Could not determine the latest available run using delay. Falling back to previous day's 18Z run.")
    target_run_hour_int = 18
    target_run_datetime_obj = now_utc.replace(hour=18, minute=0, second=0, microsecond=0) - timedelta(days=1)

# Use these in URL construction
run_hour_str = f"{target_run_hour_int:02d}" # "00", "06", "12", or "18"
run_datetime_str = target_run_datetime_obj.strftime("%Y%m%d%H") # YYYYMMDDHH format

print(f"Selected forecast run for download: {run_datetime_str} (Run hour identifier: {run_hour_str})")


# --- Forecast Horizon ---
# RDPS provides up to 84 hours. Let's fetch 48 hours forecast (P000 to P048) = 49 steps
forecast_range = 49

# --- Variable Mapping & Filename Patterns ---
# **CRITICAL**: YOU MUST VERIFY these variable codes (keys) and full filenames
# against the ECCC Datamart for the RDPS 10km model for your selected run.
# Mismatches = 404 errors. Check here: https://dd.weather.gc.ca/model_gem_regional/10km/grib2/{run_hour_str}/
# Using UGRD/VGRD for wind is standard. APCP is accumulated. VIS is often problematic.
variable_patterns = {
    # Variable Code: Filename Pattern
    "TMP":  f"CMC_reg_TMP_TGL_2_ps10km_{run_datetime_str}_P{{fh}}.grib2",  # Temp @ 2m
    "DPT":  f"CMC_reg_DPT_TGL_2_ps10km_{run_datetime_str}_P{{fh}}.grib2",  # Dew Point @ 2m
    "RH":   f"CMC_reg_RH_TGL_2_ps10km_{run_datetime_str}_P{{fh}}.grib2",   # Rel Hum @ 2m
    "UGRD": f"CMC_reg_UGRD_TGL_10_ps10km_{run_datetime_str}_P{{fh}}.grib2", # U-Wind @ 10m
    "VGRD": f"CMC_reg_VGRD_TGL_10_ps10km_{run_datetime_str}_P{{fh}}.grib2", # V-Wind @ 10m
    "APCP": f"CMC_reg_APCP_SFC_0_ps10km_{run_datetime_str}_P{{fh}}.grib2",  # Accumulated Precip @ Surface
    #"VIS":  f"CMC_reg_VIS_SFC_0_ps10km_{run_datetime_str}_P{{fh}}.grib2", # Visibility - Uncomment & VERIFY if needed
}

# --- Input/Output Files ---
# TODO: Make sure this path is correct and the file exists & has the right columns
stations_file = "Cleaned_Closest_Representative_Stations.csv"
output_csv = f"rdps_forecast_{run_datetime_str}_extracted_{forecast_range-1}h.csv" # Pivoted output
raw_output_csv = f"rdps_download_log_{run_datetime_str}.csv" # Log of download attempts
temp_filename = "temp_download.grib2" # Temporary file for download

# =======================
# READ STATION DATA
# =======================
print(f"Reading station data for coordinates from: {stations_file}")
try:
    stations_df = pd.read_csv(stations_file)
    # TODO: Verify these column names match your CSV file exactly
    required_cols = ["Region", "Station_Latitude", "Station_Longitude", "Closest_Station_ID", "Closest_Station_Name"] # Add station ID/Name if needed for reference
    if not all(col in stations_df.columns for col in ["Region", "Station_Latitude", "Station_Longitude"]):
        raise ValueError(f"Station file must contain at least columns: Region, Station_Latitude, Station_Longitude")

    stations = stations_df.rename(columns={
        "Region": "region",
        "Station_Latitude": "lat",
        "Station_Longitude": "lon"
        # Add "Closest_Station_ID": "station_id" etc. if needed
    }).to_dict('records')
    print(f"Loaded coordinates for {len(stations)} stations/regions.")
except FileNotFoundError:
    print(f"ERROR: Station file not found at {stations_file}")
    exit()
except Exception as e:
    print(f"ERROR: Failed to read or process station file: {e}")
    exit()

# =======================
# MAIN DATA RETRIEVAL AND EXTRACTION LOOP
# =======================
output_data = []       # List to store successfully extracted forecast values per station/hour/var
download_log = []      # List to log status of each download attempt

print(f"\nStarting data retrieval for run {run_datetime_str} (Hours 0 to {forecast_range-1})...")

# Loop over forecast hours
for fh_int in tqdm(range(forecast_range), desc="Processing forecast hours"):
    fh_str = f"{fh_int:03}" # Format as 3 digits (e.g., "000", "001")

    # Loop over desired variables
    for var_code, pattern in variable_patterns.items():
        # Build filename and URL
        file_name = pattern.format(fh=fh_str)
        file_url = f"{model_base_url}/{run_hour_str}/{fh_str}/{file_name}"
        log_entry = {
            "forecast_hour": fh_int,
            "variable_req": var_code,
            "filename": file_name,
            "url": file_url,
            "status": "Attempting Download",
            "error_details": "",
            "grib_vars_found": []
        }
        # print(f"  Attempting: {file_url}") # Uncomment for verbose URL checking

        # --- Download GRIB File ---
        response = None
        try:
            response = requests.get(file_url, timeout=30) # Slightly longer timeout
            log_entry["status"] = f"HTTP_{response.status_code}"
            # print(f"    Status Code: {response.status_code}") # Uncomment for status code debugging
            response.raise_for_status() # Check for HTTP errors (like 404 Not Found)

            with open(temp_filename, "wb") as f:
                f.write(response.content)
            log_entry["status"] = "Downloaded"

            # --- Open and Process GRIB File ---
            ds = None
            actual_var_name = None # Reset for each file
            try:
                # Open dataset using cfgrib, silencing the FutureWarning
                ds = xr.open_dataset(
                    temp_filename,
                    engine="cfgrib",
                    backend_kwargs={'decode_timedelta': False} # Silence warning
                )
                log_entry["status"] = "Opened_GRIB"
                log_entry["grib_vars_found"] = list(ds.data_vars)

                # --- Determine actual variable name inside GRIB ---
                # Tries requested code first, then common ECCC names, then first var if only one
                possible_names = [var_code.lower(), 't2m', 'd2m', 'r2', 'u10', 'v10', 'tp', 'vis']
                for name in possible_names:
                    if name in ds.data_vars:
                         actual_var_name = name
                         break
                if not actual_var_name and len(ds.data_vars) == 1:
                    actual_var_name = list(ds.data_vars)[0]

                # --- Check if a variable name was actually found ---
                if actual_var_name:
                    # --- Check longitude range ONCE per file ---
                    is_0_360 = False # Default
                    if 'longitude' in ds.coords:
                        min_lon = ds['longitude'].min().item()
                        max_lon = ds['longitude'].max().item()
                        # print(f"  DEBUG: GRIB Longitude Range: {min_lon:.2f} to {max_lon:.2f}") # Uncomment if needed
                        is_0_360 = max_lon > 180.0
                    else:
                        print(f"  WARNING: 'longitude' coordinate not found in {file_name}!")
                    # ---------------------------------------------

                    # --- Loop through stations and extract data ---
                    log_entry["status"] = "Extracting_Data" # Initial status for this phase
                    extraction_errors = 0 # Count errors for this file
                    for station in stations:
                        station_value = np.nan # Default to NaN
                        try:
                            # --- Prepare longitude for interpolation ---
                            station_lon_for_interp = station["lon"]
                            if is_0_360 and station_lon_for_interp < 0:
                                station_lon_for_interp += 360
                            # -----------------------------------------

                            # print(f"    Processing Station: {station.get('region', 'N/A')}, Var: {actual_var_name}, Coords: ({station['lat']},{station_lon_for_interp})") # DEBUG

                            # Interpolate using coordinate names 'latitude' and 'longitude'
                            sel_data = ds[actual_var_name].interp(
                                y=station["lat"],
                                x=station_lon_for_interp, # Use potentially adjusted longitude
                                method="nearest"
                            )
                            # print(f"      Selected data type: {type(sel_data)}, shape: {sel_data.shape}, value(s): {sel_data.values}") # DEBUG

                            # Extract scalar value safely
                            station_value = float(sel_data.item())
                            # print(f"      Successfully extracted value: {station_value}") # DEBUG

                        except Exception as extract_error:
                            # Print only the first extraction error per file for brevity
                            if extraction_errors == 0:
                                print(f"    ⚠️ FAILED first extraction for {actual_var_name} in {file_name} at station {station.get('region', 'N/A')}: {extract_error}")
                            extraction_errors += 1
                            # Keep value as NaN implicitly

                        # Append result to output list
                        output_data.append({
                            "region": station.get("region", "Unknown"),
                            "lat": station.get("lat", np.nan),
                            "lon": station.get("lon", np.nan),
                            "forecast_hour": fh_int,
                            "variable": var_code, # Use original requested code
                            # "grib_variable": actual_var_name, # Optional: store grib name found
                            "value": station_value
                        })
                    # Update status after looping through stations
                    if extraction_errors == 0:
                        log_entry["status"] = "Extraction_Complete"
                    else:
                         log_entry["status"] = f"Extraction_Partial ({extraction_errors} errors)"
                         log_entry["error_details"] = f"See console output for first extraction error detail."

                # --- Handle case where no suitable variable name was found ---
                else:
                    log_entry["status"] = "GRIB_Var_Not_Found"
                    log_entry["error_details"] = f"Could not identify target var from {list(ds.data_vars)}"
                    print(f"  ⚠️ {log_entry['status']} for {file_name}: {log_entry['error_details']}")

            except Exception as open_error:
                log_entry["status"] = "GRIB_Open_Error"
                log_entry["error_details"] = str(open_error)
                print(f"  ⚠️ Error opening/processing {file_name}: {open_error}")

            finally:
                if ds is not None:
                    ds.close() # Close the xarray dataset

        except requests.exceptions.RequestException as req_error:
            log_entry["status"] = log_entry.get("status", "Download_Error") # Keep HTTP status if available
            log_entry["error_details"] = str(req_error)
            # print(f"  ⚠️ Download failed for {file_name}: {req_error}") # Uncomment for details

        finally:
             # Delete the temporary file regardless of success/failure in processing
             if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    print(f"  Warning: Could not remove temporary file {temp_filename}: {e}")

        # Append log entry after processing each file attempt
        download_log.append(log_entry)
        # time.sleep(0.1) # Optional small delay between variable requests

# =======================
# SAVE LOG AND RESULTS
# =======================
print("\nSaving download log...")
df_log = pd.DataFrame(download_log)
df_log.to_csv(raw_output_csv, index=False, encoding='utf-8')
print(f"✅ Download log saved to {raw_output_csv}")
print("   Inspect log for HTTP_404 or other errors.")

print("\nAggregating and saving extracted forecast data...")
if not output_data:
    print("WARNING: No data was successfully extracted. Check download log and script output.")
else:
    df_results = pd.DataFrame(output_data)

    # Pivot the table - Creates columns for each variable code (TMP, DPT, etc.)
    try:
        # Check for sufficient data before pivoting
        if df_results['value'].isnull().all():
             print("WARNING: All extracted values are NaN. Pivot table will be empty or contain only NaNs.")
             # Save the raw data instead, as pivot will be meaningless
             raise ValueError("All extracted values were NaN.")

        df_pivot = df_results.pivot_table(
            index=["region", "lat", "lon", "forecast_hour"],
            columns="variable", # Use original requested code for columns
            values="value"
        ).reset_index()

        # Clean up column names if needed (MultiIndex -> single level)
        df_pivot.columns.name = None

        # Optional: Rename columns for clarity
        column_rename_map = {'TMP': 'Temperature', 'DPT': 'DewPoint', 'RH': 'Humidity',
                             'UGRD': 'U_Wind', 'VGRD': 'V_Wind', 'APCP': 'Precip_Accum'}
        df_pivot = df_pivot.rename(columns=column_rename_map)

        # Save to CSV
        df_pivot.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ Pivoted forecast data saved to {output_csv}")
        print(f"   Output shape: {df_pivot.shape}")
        print("\nNOTE: 'Precip_Accum' column contains *accumulated* precipitation since forecast start.")
        print("      You need to difference consecutive hours (df_pivot['Precip_Accum'].diff()) to get hourly rates,")
        print("      potentially grouping by region/lat/lon first before differencing.")

    # Handle errors during pivoting, likely due to missing variables or all NaNs
    except KeyError as e:
         print(f"ERROR during pivot: Key error, likely missing expected variable column after extraction. Check log/raw data. Error: {e}")
         print("Saving raw (un-pivoted) extracted data instead for inspection.")
         df_results.to_csv(output_csv.replace(".csv", "_raw_extracted.csv"), index=False, encoding='utf-8')
         print(f"✅ Raw extracted data saved to {output_csv.replace('.csv', '_raw_extracted.csv')}")
    except Exception as e:
         print(f"ERROR during final processing/saving: {e}")
         print("Saving raw (un-pivoted) extracted data instead for inspection.")
         df_results.to_csv(output_csv.replace(".csv", "_raw_extracted.csv"), index=False, encoding='utf-8')
         print(f"✅ Raw extracted data saved to {output_csv.replace('.csv', '_raw_extracted.csv')}")

print("\nScript finished.")
# --- End of Script ---