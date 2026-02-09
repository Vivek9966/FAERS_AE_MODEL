import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from validation_cases import validation_cases


RESULTS_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/results_private")

class PRRCalculator:
    def __init__(self , pairs_df):
        self.pairs_df = pairs_df
        self.total_reports = len(pairs_df)
        self.drug_count= pairs_df.groupby('drug').size().to_dict()
        self.event_count = pairs_df.groupby('event').size().to_dict()
        self.DEcounts = pairs_df.groupby(['drug','event']).size().to_dict()

    def calculate_prr(self,drug,event):
        a = self.DEcounts.get((drug,event),0)
        
        if a == 0:
            return None
            
        total_d = self.drug_count.get(drug,0)
        b = total_d-a
        total_e = self.event_count.get(event,0)
        c = total_e-a
        d = self.total_reports -a -b -c

        if b<= 0 or c <= 0 or d<=0:
            return None
            
        prr= (a*d)/(b*c)
        cont_table = np.array([[a,b],[c,d]])
        chi,p_val,dof,excepted = stats.chi2_contingency(cont_table)
        signal_ = (prr>2) and (chi>4) and (a>=3)

        return {
            'drug':drug,
            'event':event, 
            'n_cases':a,
            'n_drug':total_d,
            'n_event':total_e, 
            'prr':prr,
            'chi2':chi,
            'p_value':p_val,
            'dof':dof,
            'is_signal':signal_,
            'a':a,
            'b':b,
            'c':c,
            'd':d
        }
        
    def prr_validation(self,cases):
        results = []
        for i , case in enumerate(cases,1):
            drug = case['drug']
            event = case['event']

            drug_ = [d for d in self.drug_count.keys() if drug.upper() in d.upper()] 
            if not drug_:
                print(f"  Drug not found: {drug}")
                continue
            drug__ = max(drug_,key = lambda d: self.drug_count[d])
            result = self.calculate_prr(drug__,event)

            if result:
                result['expected_prr'] = case.get('expected_prr', 'unknown')
                result['description'] = case.get('description', '')
                results.append(result)
                print(f"  ✓ {drug__} + {event}: PRR={result['prr']:.2f}, n={result['n_cases']}")
            else:
                print(f"  ✗ No data for: {drug} + {event}")
                
        return pd.DataFrame(results)

    def cal_signals(self,top_dr = 100 , top_eve = 500 , min_cases = 3):
        top_drugs = sorted(self.drug_count.keys(),key = lambda x: self.drug_count[x], reverse =True)[:top_dr]
        top_events = sorted(self.event_count.keys(), key = lambda x: self.event_count[x] ,reverse=True)[:top_eve]
        results = []
        total = len(top_drugs)*len(top_events)
        
       
        for i, drug in enumerate(top_drugs,1):
           
            
            for event in top_events:
                result = self.calculate_prr(drug,event)
                if result and result['n_cases']>= min_cases:
                    results.append(result)
        
        print(f"\n  Completed: {len(results)} signals found")
        df = pd.DataFrame(results)
        return df.sort_values('prr',ascending = False)

validation_cases = validation_cases()
print(f"  Loaded {len(validation_cases)} validation cases")


drug_event = pd.read_parquet("/home/vivekbisht/Desktop/faers-signal-detection/results_private/drug_event_pairs.parquet")
print(f"  Loaded {len(drug_event):,} pairs")

prr = PRRCalculator(drug_event)
print(f"  {len(prr.drug_count):,} unique drugs")
print(f"  {len(prr.event_count):,} unique events")


validated = prr.prr_validation(validation_cases)

if len(validated) > 0:
    validated.to_csv("/home/vivekbisht/Desktop/faers-signal-detection/results_private/prr_validation_results.csv",index=False)
    print(validated[['drug','event','n_cases','prr','is_signal','expected_prr']].to_string(index=False))


OUTPUT = RESULTS_DIR / 'baseline_prr.parquet'
top_signals = prr.cal_signals()

if len(top_signals) > 0:
    top_signals.to_parquet(OUTPUT, index=False)

    signals_only = top_signals[top_signals['is_signal'] == True]
    print(f"\nTotal signals (PRR≥2, chi2≥4, n≥3): {len(signals_only):,}")

    print(signals_only.head(20)[['drug','event','prr','n_cases','chi2']].to_string(index=False))

