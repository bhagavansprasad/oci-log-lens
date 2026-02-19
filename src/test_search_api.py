#!/usr/bin/env python3
"""
test_search_api.py
------------------
Test the /search endpoint with a real log file.
"""

import json
import requests

# Read log file
with open('flow-logs/01_flow-log.json', 'r') as f:
    log_content = f.read()

# Call search endpoint
print("=== Searching for similar logs ===")
response = requests.post(
    'http://localhost:8000/search',
    json={'log_content': log_content}
)

# Print results
print(f"\nStatus Code: {response.status_code}\n")

if response.status_code == 200:
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"Message: {data['message']}\n")
    
    for i, match in enumerate(data['matches'], 1):
        print(f"Rank {i}:")
        print(f"  Jira ID    : {match['jira_id']}")
        print(f"  Similarity : {match['similarity_score']}%")
        print(f"  Flow Code  : {match['flow_code']}")
        print(f"  Trigger    : {match['trigger_type']}")
        print(f"  Error Code : {match['error_code']}")
        print(f"  Summary    : {match['error_summary']}")
        print()
else:
    print(f"Error: {response.json()}")