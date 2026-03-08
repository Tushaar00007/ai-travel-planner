import pandas as pd
import os

# Define the path to the dataset
csv_path = '/Users/tushaarrohatgi/Developer/planner/Ml_model/dataset/cleaned_pan_india_tourism_dataset_updated.csv'

if not os.path.exists(csv_path):
    print(f"Error: Dataset not found at {csv_path}")
    exit(1)

# Load the dataset
df = pd.read_csv(csv_path)

# Comprehensive mapping for major Indian cities/states with multiple or updated airports
AIRPORT_MAPPING = {
    "Goa": "Dabolim Airport / Manohar International Airport (Mopa)",
    "Delhi": "Indira Gandhi International Airport (IGI) / Hindon Airport",
    "New Delhi": "Indira Gandhi International Airport (IGI) / Hindon Airport",
    "Mumbai": "Chhatrapati Shivaji Maharaj International Airport (CSMIA)",
    "Mumbai Suburban": "Chhatrapati Shivaji Maharaj International Airport (CSMIA)",
    "Bangalore": "Kempegowda International Airport",
    "Bengaluru": "Kempegowda International Airport",
    "Hyderabad": "Rajiv Gandhi International Airport",
    "Chennai": "Chennai International Airport",
    "Kolkata": "Netaji Subhas Chandra Bose International Airport",
    "Ahmedabad": "Sardar Vallabhbhai Patel International Airport",
    "Pune": "Pune International Airport",
    "Lucknow": "Chaudhary Charan Singh International Airport",
    "Varanasi": "Lal Bahadur Shastri International Airport",
    "Jaipur": "Jaipur International Airport",
    "Amritsar": "Sri Guru Ram Dass Jee International Airport",
    "Kochi": "Cochin International Airport",
    "Cochin": "Cochin International Airport",
    "Thiruvananthapuram": "Trivandrum International Airport",
    "Bhubaneswar": "Biju Patnaik International Airport",
    "Guwahati": "Lokpriya Gopinath Bordoloi International Airport",
    "Indore": "Devi Ahilya Bai Holkar Airport",
    "Chandigarh": "Chandigarh International Airport / Shaheed Bhagat Singh International Airport",
    "Patna": "Jay Prakash Narayan International Airport",
    "Noida": "Indira Gandhi International Airport / Noida International Airport (Jewar - Upcoming)",
}

# Function to update airport column based on City or State
def update_airport(row):
    city = str(row['City']).strip()
    state = str(row['State']).strip()
    
    # Check City first, then State
    if city in AIRPORT_MAPPING:
        return AIRPORT_MAPPING[city]
    if state in AIRPORT_MAPPING:
        return AIRPORT_MAPPING[state]
    
    # If already has a valid name, keep it, otherwise return original
    current = str(row['Nearest Airport'])
    if current in ['Unknown', 'Not Available', 'None', 'nan', '']:
        return current
    return current

# Apply the mapping
df['Nearest Airport'] = df.apply(update_airport, axis=1)

# Special handling for Goa specifically as requested
df.loc[df['State'].str.contains('Goa', case=False, na=False), 'Nearest Airport'] = AIRPORT_MAPPING["Goa"]

# Also update "Major Railway Station" placeholders if missing for Goa
df.loc[(df['State'].str.contains('Goa', case=False, na=False)) & (df['Major Railway Station'] == 'Unknown'), 'Major Railway Station'] = "Madgaon Junction / Karmali / Thivivm"

# Save the updated dataset
df.to_csv(csv_path, index=False)
print("Dataset enriched successfully with updated airport information.")
