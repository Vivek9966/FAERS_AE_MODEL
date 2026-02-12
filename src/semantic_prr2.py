import pandas as pd
from pathlib import Path
from semantic_prr import SemanticEnhancedPRR

RESULTS_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/results_private")
pairs = pd.read_parquet(RESULTS_DIR / 'drug_event_pairs.parquet')

semantic_prr = SemanticEnhancedPRR(pairs)

rare_event_cases = [
    ('ISOTRETINOIN', 'SUICIDAL IDEATION'),     
    ('CLOZAPINE', 'LEUKOPENIA'),                
    ('WARFARIN', 'INTRACRANIAL HAEMORRHAGE'),   
    ('METFORMIN', 'METABOLIC ACIDOSIS'),       
    ('FINGOLIMOD', 'ATRIOVENTRICULAR BLOCK'),   
]

for drug, event in rare_event_cases:
    result = semantic_prr.calculate_semantic_enhanced_prr(drug, event, top_k=5, min_similarity=0.80)
    
    if result:
        print(f"\n{drug} + {event}")
        print(f"  Baseline PRR:  {result['prr']:.2f} (n={result['n_cases']})")
        print(f"  Enhanced PRR:  {result['enhanced_prr']:.2f}")
        print(f"  Improvement:   {result['improvement']:+.1f}%")
        print(f" Similar events: {result['similar_events_used']}")
        
        if result['similar_events_used'] > 0:
            for se in result['similar_events'][:3]:
                print(f"{se['event'][:40]:40s} sim={se['similarity']:.3f} PRR={se['prr']:.2f} n={se['n_cases']}")