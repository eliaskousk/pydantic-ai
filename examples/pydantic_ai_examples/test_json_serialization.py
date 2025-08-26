"""Test to reproduce the JSON serialization issue with LatLng objects."""

import json
from weather_agent import LatLng

# Create a LatLng instance
lat_lng = LatLng(lat=51.5074, lng=-0.1278)

# This will raise: TypeError: Object of type LatLng is not JSON serializable
try:
    json.dumps(lat_lng)
    print("ERROR: Should have raised TypeError")
except TypeError as e:
    print(f"Expected error: {e}")

# Solution 1: Use Pydantic's model_dump()
print(f"Solution 1 - model_dump(): {json.dumps(lat_lng.model_dump())}")

# Solution 2: Use Pydantic's model_dump_json()
print(f"Solution 2 - model_dump_json(): {lat_lng.model_dump_json()}")