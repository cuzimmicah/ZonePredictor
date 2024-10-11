import requests
import json
import random
import os
from dotenv import load_dotenv

# load environmental variables
load_dotenv()

# Ensure the data directory exists
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
api_key = '48268bce-0ba6-492f-8597-c04db56622e2'

# Function to fetch match IDs and save them to a JSON file
def get_match_ids(api_key, filename=f'{data_dir}/match_ids.json'):
    url = "https://api.osirion.gg/fortnite/v1/matches?epicIds=256ebe483e954f1fb53cb3f9d4b757aa"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        new_match_ids = extract_match_ids(data)
        
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                try:
                    existing_match_ids = json.load(file)
                except json.JSONDecodeError:
                    existing_match_ids = []
        else:
            existing_match_ids = []
        
        combined_match_ids = list(set(existing_match_ids + new_match_ids))
        
        with open(filename, 'w') as file:
            json.dump(combined_match_ids, file, indent=4)
        
        print(f"Match IDs appended to {filename}")
    else:
        print(f"Failed to fetch match IDs. Status code: {response.status_code}")

# Helper function to extract match IDs from the API response
def extract_match_ids(data):
    TWENTYFIVEMINS = 1500000
    match_ids = []
    for match in data.get('matches', []):
        match_id = match.get('info', {}).get('matchId')
        length_ms = match.get('info', {}).get('lengthMs')

        # 25 Minutes is more than the length of a full fortnite game
        if length_ms > TWENTYFIVEMINS:
            match_ids.append(match_id)
    return match_ids

# Helper function to save data to a JSON file
def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Function to fetch match data for a given number of random match IDs
def get_match_data(api_key, num_matches, input_filename=f'{data_dir}/match_ids.json', output_filename=f'{data_dir}/match_data.json'):
    with open(input_filename, 'r') as json_file:
        match_ids = json.load(json_file)
    
    match_data = []
    attempts = 0
    max_attempts = num_matches * 2  # Allow some room for failures
    loops = 0

    while len(match_data) < num_matches and attempts < max_attempts:
        attempts += 1
        match_id = random.choice(match_ids)

        url = f"https://api.osirion.gg/fortnite/v1/matches/{match_id}/events?include=safeZoneUpdateEvents"
        headers = {
            'Authorization': f'Bearer {api_key}'
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            match_data.append(response.json())
        else:
            print(f"Failed to fetch match data for {match_id}. Status code: {response.status_code}")
        print(loops)
        loops = loops + 1
    
    save_to_json(match_data, output_filename)
    print(f"Match data saved to {output_filename}")

# Function to extract and format zone data for LSTM model usage
def extract_zone_data(match_data):
    all_zone_data = []
    zone_radii = [120000, 95000, 70000, 55000, 32500, 20000, 10000, 5000, 2500, 1650, 1090, 0]

    # Pre-create match_zones template for all matches to avoid redundant creation within the loop
    match_zones_template = [{'currentPhase': phase, 'zoneRadius': zone_radii[phase - 1]} for phase in range(1, 13)]

    for match in match_data:
        safe_zone_events = match.get('safeZoneUpdateEvents', [])
        match_zones = [dict(zone) for zone in match_zones_template]
        
        for event in safe_zone_events:
            current_phase = event.get('currentPhase')
            next_center = event.get('nextCenter', {})
  
            center = {'x': next_center.get('x'), 'y': next_center.get('y')}
            match_zones[current_phase - 1]['center'] = center

        all_zone_data.append({'zones': match_zones})
    
    return all_zone_data


# Function to run the entire process
def run_process(api_key, num_matches_to_fetch, need_matches=True):
    # Step 1: Fetch and save match IDs
    if need_matches:
        get_match_ids(api_key)

    # Step 2: Fetch and save match data
    get_match_data(api_key, num_matches=num_matches_to_fetch)

    # Load the fetched match data
    with open(f'{data_dir}/match_data.json', 'r') as file:
        match_data = json.load(file)

    # Step 3: Extract and save zone data
    zone_data = extract_zone_data(match_data)
    save_to_json(zone_data, f'{data_dir}/extracted_zone_data.json')
    print(f"Extracted zone data saved to {data_dir}/extracted_zone_data.json")

# Example usage: Run the entire process
run_process(api_key, num_matches_to_fetch=1000, need_matches=False)
