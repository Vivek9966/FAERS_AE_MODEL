import pandas as pd
from pathlib import Path


VALIDATION_CASES = [
    {
        'drug': 'ISOTRETINOIN',
        'event': 'DEPRESSION',
        'year_discovered': 1998,
        'description': 'Acne drug - psychiatric effects (ongoing monitoring)',
        'expected_prr': '>3'
    },
    {
        'drug': 'FLUOROQUINOLONE',
        'event': 'TENDON RUPTURE',
        'year_discovered': 2008,
        'description': 'Black box warning added',
        'expected_prr': '>5',
        'note': 'Search for LEVOFLOXACIN, CIPROFLOXACIN, MOXIFLOXACIN'
    },
    {
        'drug': 'GABAPENTIN',
        'event': 'RESPIRATORY DEPRESSION',
        'year_discovered': 2019,
        'description': 'FDA warning about respiratory risk',
        'expected_prr': '>2'
    },
    {
        'drug': 'CANAGLIFLOZIN', 
        'event': 'AMPUTATION',
        'year_discovered': 2017,
        'description': 'SGLT2 inhibitor - toe/foot amputation',
        'expected_prr': '>3'
    },
    {
        'drug': 'METFORMIN',
        'event': 'LACTIC ACIDOSIS',
        'year_discovered': 'long known',
        'description': 'Classic diabetes drug ADR',
        'expected_prr': '>5'
    },
    {
        'drug': 'WARFARIN',
        'event': 'HAEMORRHAGE',
        'year_discovered': 'long known',
        'description': 'Anticoagulant bleeding risk',
        'expected_prr': '>10'
    },
    {
        'drug': 'LITHIUM',
        'event': 'HYPOTHYROIDISM',
        'year_discovered': 'long known',
        'description': 'Mood stabilizer - thyroid effects',
        'expected_prr': '>5'
    },
    {
        'drug': 'CLOZAPINE',
        'event': 'AGRANULOCYTOSIS',
        'year_discovered': 'long known',
        'description': 'Antipsychotic - blood disorder',
        'expected_prr': '>50'
    },
    {
        'drug': 'LINEZOLID',
        'event': 'SEROTONIN SYNDROME',
        'year_discovered': 2011,
        'description': 'Antibiotic with MAOI effects',
        'expected_prr': '>10'
    },
    {
        'drug': 'FINGOLIMOD', 
        'event': 'BRADYCARDIA',
        'year_discovered': 2012,
        'description': 'MS drug - heart rate decrease',
        'expected_prr': '>5'
    }
]

def validation_cases():
    return VALIDATION_CASES
def check_validation():
    RESULTS_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/results_private")
    pairs = pd.read_parquet(RESULTS_DIR / 'drug_event_pairs.parquet')      
    found = []
    
    for case in VALIDATION_CASES:
        drug_pattern = case['drug']
        event = case['event']
        
        if 'note' in case:
            continue
        drug_matches = pairs[pairs['drug'].str.contains(drug_pattern, na=False, case=False)]
        event_matches = pairs[pairs['event'] == event]
        combo = pairs[
            (pairs['drug'].str.contains(drug_pattern, na=False, case=False)) &
            (pairs['event'] == event)
        ] 
        if len(combo) >= 3:
            found.append(case)   
    
    return found

if __name__ == "__main__":
    found =check_validation()
    print(f"Found {len(found)} out of {len(VALIDATION_CASES)} validation cases:")