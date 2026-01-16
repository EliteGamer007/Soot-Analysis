import csv
import math
import random
from datetime import datetime, timedelta, timezone

# --- CONFIG ---
SENSOR_OUTPUT_PATH = "data/sensor_telemetry.csv"
TRIP_OUTPUT_PATH = "data/trip_characteristics.csv"
MAINT_OUTPUT_PATH = "data/maintenance_records.csv"

RANDOM_SEED = 42
NUM_VEHICLES = 50
SAMPLING_MINUTES = 5  # Once every 5 mins
DAYS_OF_DATA = 30     # 1 Month duration

START_DATE = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def daily_cycle(hour):
    """Simulates daily temperature swing (Peak heat ~3 PM)."""
    return math.sin((hour - 15) / 24 * 2 * math.pi)

def generate_vehicle_params(vid):
    """Assigns vehicle personality."""
    base_mass = random.uniform(8000, 32000)
    engine_disp = random.uniform(10.0, 15.0)
    urban_bias = random.uniform(0.1, 0.8)  # 0.1=Highway, 0.8=City Delivery
    
    return {
        "vehicle_id": f"VEH-{vid:03d}",
        "base_mass": base_mass,
        "engine_disp": engine_disp,
        "urban_bias": urban_bias,
        "odometer": random.randint(10000, 500000)
    }

def synthesize():
    random.seed(RANDOM_SEED)
    total_rows = 0
    end_date = START_DATE + timedelta(days=DAYS_OF_DATA)

    sensor_header = [
        "Vehicle ID", "Timestamp", "Vehicle speed", "Engine RPM", "Engine load (%)",
        "Ambient temperature", "Fuel consumption rate", "Manifold absolute pressure",
        "Intake air temperature", "Exhaust flow rate", "Exhaust temperature - DOC inlet",
        "Exhaust temperature - DPF inlet", "Exhaust temperature - DPF outlet",
        "Differential pressure across DPF", "DPF pressure drop (or soot proxy)",
        "NOx concentration"
    ]
    
    trip_header = [
        "Vehicle ID", "Trip start", "Trip end", "Trip duration (min)",
        "Trip distance (km)", "Idle time (min)", "Stop-start count",
        "Avg speed", "Avg engine load", "Driving pattern"
    ]

    maint_header = [
        "Vehicle ID", "Maintenance timestamp", "Action type",
        "Trigger reason", "Estimated soot", "DPF differential pressure", "Notes"
    ]

    with open(SENSOR_OUTPUT_PATH, "w", newline="", encoding="utf-8") as sf, \
         open(TRIP_OUTPUT_PATH, "w", newline="", encoding="utf-8") as tf, \
         open(MAINT_OUTPUT_PATH, "w", newline="", encoding="utf-8") as mf:

        sensor_writer = csv.writer(sf)
        trip_writer = csv.writer(tf)
        maint_writer = csv.writer(mf)

        sensor_writer.writerow(sensor_header)
        trip_writer.writerow(trip_header)
        maint_writer.writerow(maint_header)

        print(f"Simulating {NUM_VEHICLES} vehicles with Progressive Idle Scaling...")

        for vid in range(1, NUM_VEHICLES + 1):
            params = generate_vehicle_params(vid)
            
            # --- State Initialization ---
            soot = random.uniform(0.8, 2.5)
            last_regen = START_DATE
            last_scheduled_maint = START_DATE - timedelta(days=random.randint(0, 5))
            
            current_mode = 'PARKED'
            mode_duration_remaining = 0
            
            # Trip tracking variables
            trip_data = {
                "start_time": None, "speed_sum": 0, "load_sum": 0, "count": 0,
                "idle_mins": 0, "stops": 0, "distance": 0, "last_speed": 0, 
                "pattern": "mixed",
                "rest_needed_mins": 0,  # Total rest required for this trip
                "rest_taken_mins": 0,   # How much we have taken so far
                "is_forcing_rest": False 
            }

            t = START_DATE
            while t < end_date:
                
                # --- 1. TRIP & MODE DECISION LOGIC ---
                if mode_duration_remaining <= 0:
                    
                    if current_mode == 'PARKED':
                        # START NEW TRIP
                        if random.random() < params["urban_bias"]:
                            # City Trip
                            current_mode = 'CITY'
                            trip_data["pattern"] = "City Delivery"
                            est_distance_km = random.randint(20, 120)
                            avg_speed_est = 25
                            drive_mins = (est_distance_km / avg_speed_est) * 60
                            # City stops are frequent but random (traffic/delivery)
                            req_rest_mins = drive_mins * random.uniform(0.1, 0.3) 

                        else:
                            # Highway / Long Haul
                            if random.random() < 0.2:
                                current_mode = 'HEAVY_LOAD'
                                trip_data["pattern"] = "Heavy Haul"
                                avg_speed_est = 60
                            else:
                                current_mode = 'HIGHWAY'
                                trip_data["pattern"] = "Highway Cruise"
                                avg_speed_est = 85
                            
                            # Determine Distance
                            if random.random() < 0.08: # Outlier (Super Long)
                                est_distance_km = random.randint(800, 1800) 
                                trip_data["pattern"] += " (Long Haul)"
                            else:
                                est_distance_km = random.randint(50, 900)

                            drive_mins = (est_distance_km / avg_speed_est) * 60
                            
                            if est_distance_km < 100:
                                # Short trips (<100km): Keep it low/random 
                                req_rest_mins = random.uniform(0, 15) 
                            else:
                                # Base: 20 mins minimum, plus scaling
                                req_rest_mins = 20 + (drive_mins * 0.15) 

                                # Tiered Add-ons for Realism
                                if est_distance_km > 300:
                                    req_rest_mins += random.randint(20, 45) # Coffee/Restroom
                                if est_distance_km > 600:
                                    req_rest_mins += random.randint(45, 70) # Lunch/Dinner break
                                if est_distance_km > 1000:
                                    req_rest_mins += random.randint(360, 540) # Mandatory Sleep (6-9 hours)

                        # Set Duration
                        total_duration = int(drive_mins + req_rest_mins)
                        mode_duration_remaining = total_duration
                        
                        # Initialize Trip Data
                        trip_data.update({
                            "start_time": t, "speed_sum": 0, "load_sum": 0, "count": 0,
                            "idle_mins": 0, "stops": 0, "distance": 0, "last_speed": 0,
                            "rest_needed_mins": req_rest_mins,
                            "rest_taken_mins": 0,
                            "is_forcing_rest": False
                        })
                        
                    else:
                        avg_spd = trip_data["speed_sum"] / max(1, trip_data["count"])
                        avg_ld = trip_data["load_sum"] / max(1, trip_data["count"])
                        trip_dur = (t - trip_data["start_time"]).total_seconds() / 60
                        
                        if trip_dur > 15 and trip_data["distance"] > 1:
                            trip_writer.writerow([
                                params["vehicle_id"], trip_data["start_time"].isoformat(),
                                t.isoformat(), int(trip_dur), round(trip_data["distance"], 1),
                                trip_data["idle_mins"], trip_data["stops"],
                                round(avg_spd, 1), round(avg_ld, 1), trip_data["pattern"]
                            ])
                        
                        current_mode = 'PARKED'
                        duration_mins = random.randint(60, 600) # Park duration
                        mode_duration_remaining = duration_mins

                # --- 2. PHYSICS & DRIVING BEHAVIOR ---
                target_speed = 0
                
                if current_mode == 'PARKED':
                    t += timedelta(minutes=SAMPLING_MINUTES)
                    mode_duration_remaining -= SAMPLING_MINUTES
                    continue 

                # --- REST ENFORCEMENT LOGIC ---
                rest_balance = trip_data["rest_needed_mins"] - trip_data["rest_taken_mins"]
                trip_progress = 1.0 - (mode_duration_remaining / ((t - trip_data["start_time"]).total_seconds()/60 + mode_duration_remaining + 1))
                
                # Check if we should be sleeping (Forced Rest)
                if trip_data["is_forcing_rest"]:
                    # Continue sleeping until balance depleted or random wake up if nearly done
                    target_speed = 0
                    if rest_balance <= 0:
                        trip_data["is_forcing_rest"] = False
                else:
                    # Decide if we SHOULD start a rest block
                    if rest_balance > 0:
                        # If huge rest needed (>2 hours) and we are >45% through trip -> FORCE SLEEP
                        if rest_balance > 120 and trip_progress > 0.45:
                            trip_data["is_forcing_rest"] = True
                            target_speed = 0
                        # If moderate rest needed (>45 mins) and we are >30% through -> MEAL BREAK
                        elif rest_balance > 45 and trip_progress > 0.3 and trip_progress < 0.7:
                            if random.random() < 0.1: # 10% chance per tick to start the meal
                                trip_data["is_forcing_rest"] = True
                                target_speed = 0
                        # Small random breaks for short stops
                        elif random.random() < 0.02: 
                            target_speed = 0
                        else:
                            # Drive normally
                            if current_mode == 'HIGHWAY': target_speed = random.uniform(75, 100)
                            elif current_mode == 'CITY': target_speed = random.uniform(10, 45)
                            elif current_mode == 'HEAVY_LOAD': target_speed = random.uniform(50, 75)
                    else:
                        # No rest needed, drive
                        if current_mode == 'HIGHWAY': target_speed = random.uniform(75, 100)
                        elif current_mode == 'CITY': target_speed = random.uniform(10, 45)
                        elif current_mode == 'HEAVY_LOAD': target_speed = random.uniform(50, 75)

                # Apply Noise
                if target_speed > 0:
                    speed = clamp(target_speed + random.gauss(0, 4), 0, 110)
                    if speed < 2: speed = 0
                else:
                    speed = 0

                # --- 3. ENGINE PHYSICS ---
                if speed == 0:
                    rpm = 600 + random.uniform(-10, 10)
                    load = 10 + random.uniform(0, 5) # Idle load
                else:
                    rpm = clamp(900 + speed * 14, 900, 2200)
                    base_load = 50 if current_mode == 'HEAVY_LOAD' else 25
                    load = clamp(base_load + 0.4 * speed + random.gauss(0, 5), 10, 100)

                # Thermals
                exhaust_flow = clamp((rpm * params["engine_disp"] * 0.035) * (0.5 + load/150), 50, 1800)
                ambient = 18 + 7 * daily_cycle(t.hour + t.minute/60)
                
                # Exhaust cools down during those long idle sleeps
                base_temp = 160 + 3.2 * load + 0.04 * rpm
                doc_inlet = clamp(base_temp + random.gauss(0, 10), 100, 700) 

                # --- 4. SOOT & MAINTENANCE ---
                
                # Accumulate
                soot_rate = 0.005 * (load/100)
                if current_mode == 'CITY': soot_rate *= 1.8 
                
                # Idle Clogging
                if speed == 0 and doc_inlet < 200: soot_rate = 0.001 

                # Passive Regen
                if doc_inlet > 420: soot_rate -= 0.003
                
                soot += soot_rate
                soot = clamp(soot, 0.5, 8.0) 

                # Maintenance Events
                maint_type = None
                maint_reason = None
                maint_note = ""

                # Scheduled
                if (t - last_scheduled_maint).days >= 7:
                    maint_type = "inspection"
                    maint_reason = "scheduled_interval"
                    maint_note = "Routine fleet maintenance check"
                    last_scheduled_maint = t
                # Spot Check
                elif random.random() < 0.0001: 
                    maint_type = "inspection"
                    maint_reason = "driver_report"
                    maint_note = "Driver reported warning light (False Alarm)"

                # Active Regen
                is_active_regen = False
                if soot > 5.5: 
                    is_active_regen = True
                    maint_type = "active"
                    maint_reason = "critical_soot_load"
                    maint_note = "Forced regeneration (Critical)"
                elif soot > 4.2 and (t - last_regen).days > 2 and speed > 60:
                    is_active_regen = True
                    maint_type = "active"
                    maint_reason = "preventative_threshold"
                    maint_note = "Standard active regeneration cycle"

                if is_active_regen:
                    dpf_inlet = doc_inlet + random.uniform(200, 300)
                    soot = max(0.5, soot - 1.5)
                    last_regen = t
                    
                    if not maint_reason or maint_type == "active":
                         maint_writer.writerow([
                            params["vehicle_id"], t.isoformat(),
                            "active_regeneration", maint_reason or "ecu_trigger",
                            round(soot, 2), 
                            round((0.005 + 0.002 * soot) * exhaust_flow, 2),
                            maint_note or "Active regen cycle"
                        ])
                elif maint_type:
                     maint_writer.writerow([
                        params["vehicle_id"], t.isoformat(),
                        maint_type, maint_reason,
                        round(soot, 2), 
                        round((0.005 + 0.002 * soot) * exhaust_flow, 2),
                        maint_note
                    ])

                # --- 5. WRITE SENSORS ---
                if not is_active_regen:
                    dpf_inlet = doc_inlet - random.uniform(10, 30)
                dpf_outlet = dpf_inlet - random.uniform(10, 50)
                dpf_dp = clamp((0.005 + 0.002 * soot) * exhaust_flow, 0.2, 15.0) + random.gauss(0, 0.1)

                fuel_rate = (load * params["engine_disp"] * 0.04) + 1.5
                map_kpa = 100 + load * 1.8
                nox = 50 + 5 * load + 0.1 * doc_inlet

                sensor_writer.writerow([
                    params["vehicle_id"], t.isoformat(),
                    round(speed, 1), int(rpm), round(load, 1),
                    round(ambient, 1), round(fuel_rate, 2),
                    int(map_kpa), int(ambient + 15),
                    int(exhaust_flow), int(doc_inlet),
                    int(dpf_inlet), int(dpf_outlet),
                    round(dpf_dp, 2), round(soot * 10, 1), int(nox)
                ])

                # Update Trip Stats
                trip_data["count"] += 1
                trip_data["speed_sum"] += speed
                trip_data["load_sum"] += load
                trip_data["distance"] += speed * (SAMPLING_MINUTES / 60)
                
                if speed == 0:
                    trip_data["idle_mins"] += SAMPLING_MINUTES
                    trip_data["rest_taken_mins"] += SAMPLING_MINUTES
                    if trip_data["last_speed"] > 5: trip_data["stops"] += 1
                
                trip_data["last_speed"] = speed
                
                # Advance Time
                total_rows += 1
                t += timedelta(minutes=SAMPLING_MINUTES)
                mode_duration_remaining -= SAMPLING_MINUTES

    return total_rows

if __name__ == "__main__":
    rows = synthesize()
    print(f"Done! Generated {rows:,} rows.")
    print(f"Data saved to: {SENSOR_OUTPUT_PATH}, {TRIP_OUTPUT_PATH}, {MAINT_OUTPUT_PATH}")