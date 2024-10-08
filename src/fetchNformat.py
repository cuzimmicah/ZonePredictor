import requests
import json
import random
import os
from dotenv import load_dotenv

# load environmental variabels
load_dotenv()

# Ensure the data directory exists
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Function to fetch match IDs and save them to a JSON file
def get_match_ids(api_key, filename=f'{data_dir}/match_ids.json'):
    url = "https://api.osirion.gg/fortnite/v1/matches?epicIds=1d6a200529cf49febcfd1f628741f419"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        match_ids = extract_match_ids(data)
        save_to_json(match_ids, filename)
        print(f"Match IDs saved to {filename}")
    else:
        print(f"Failed to fetch match IDs. Status code: {response.status_code}")

# Helper function to extract match IDs from the API response
def extract_match_ids(data):
    match_ids = []
    for match in data.get('matches', []):
        match_id = match.get('info', {}).get('matchId')
        length_ms = match.get('info', {}).get('lengthMs')
        if match_id and length_ms and length_ms > 1380000:
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

    while len(match_data) < num_matches and attempts < max_attempts:
        attempts += 1
        match_id = random.choice(match_ids)
        match_info = fetch_match_data(api_key, match_id)
        if match_info:
            match_data.append(match_info)
            match_ids.remove(match_id)  # Remove the used ID to avoid duplicate requests
    
    save_to_json(match_data, output_filename)
    print(f"Match data saved to {output_filename}")

# Helper function to fetch match data for a specific match ID
def fetch_match_data(api_key, match_id):
    url = f"https://api.osirion.gg/fortnite/v1/matches/{match_id}/events?include=safeZoneUpdateEvents"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data for match ID {match_id}. Status code: {response.status_code}")
        return None

# Function to extract zone data from the match data
def extract_zone_data(match_data, exclude_phases=[]):
    all_zone_data = []
    excluded_zone_data = []
    
    for match in match_data:
        safe_zone_events = match.get('safeZoneUpdateEvents', [])
        
        # Collect zone data for each match
        match_zones = []
        excluded_zones = []
        for event in safe_zone_events:
            zone_info = {
                'currentPhase': event.get('currentPhase'),
                'previousRadius': event.get('previousRadius'),
                'nextRadius': event.get('nextRadius'),
                'previousCenter': event.get('previousCenter')
            }
            
            if event.get('currentPhase') in exclude_phases:
                excluded_zones.append(zone_info)
            else:
                match_zones.append(zone_info)
        
        all_zone_data.append({
            'zones': match_zones
        })
        
        if excluded_zones:
            excluded_zone_data.append({
                'matchId': event.get('matchId'),
                'excludedZones': excluded_zones
            })
    
    return all_zone_data, excluded_zone_data

# Function to run the entire process
def run_process(api_key, num_matches_to_fetch, exclude_phases=[], need_matches=True):
    # Step 1: Fetch and save match IDs
    if need_matches:
        get_match_ids(api_key)

    # Step 2: Fetch and save match data
    get_match_data(api_key, num_matches=num_matches_to_fetch)

    # Load the fetched match data
    with open(f'{data_dir}/match_data.json', 'r') as file:
        match_data = json.load(file)

    # Step 3: Extract and save zone data
    zone_data, excluded_zone_data = extract_zone_data(match_data, exclude_phases=exclude_phases)
    save_to_json(zone_data, f'{data_dir}/extracted_zone_data.json')
    save_to_json(excluded_zone_data, f'{data_dir}/excluded_zone_data.json')
    print(f"Extracted zone data saved to {data_dir}/extracted_zone_data.json")
    print(f"Excluded zone data saved to {data_dir}/excluded_zone_data.json")

api_key = os.getenv("OSIRIONKEY")

# Example usage: Run the entire process
run_process(api_key, num_matches_to_fetch=250, exclude_phases=[8,9,10,11,12], need_matches=False)
