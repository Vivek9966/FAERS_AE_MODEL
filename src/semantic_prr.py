
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import faiss

RESULTS_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/results_private")
EMBEDDINGS_FILE = RESULTS_DIR / 'ae_embeddings_sapbert.npy'
UNIQUE_AES_FILE = RESULTS_DIR / 'unique_aes.txt'

class SemanticEnhancedPRR:
    def __init__(self, pairs_df, embeddings_path=EMBEDDINGS_FILE, unique_aes_path=UNIQUE_AES_FILE):
        
        self.pairs = pairs_df
        self.total_reports = len(pairs_df)
        
        self.drug_counts = pairs_df.groupby('drug').size().to_dict()
        self.event_counts = pairs_df.groupby('event').size().to_dict()
        self.combo_counts = pairs_df.groupby(['drug', 'event']).size().to_dict()
        
        ae_embeddings_array = np.load(embeddings_path)
        
        with open(unique_aes_path, 'r', encoding='utf-8') as f:
            unique_aes_list = [line.strip() for line in f if line.strip()]
        
        
        self.event_embeddings = {}
        for ae_term, emb in zip(unique_aes_list, ae_embeddings_array):
            ae_upper = ae_term.upper()
            self.event_embeddings[ae_upper] = emb
        
        missing_events = set(self.event_counts.keys()) - set(self.event_embeddings.keys())
        if missing_events:
            if len(missing_events) <= 10:
                print(f"Warning: Missing embeddings for events: {missing_events}")
    
    def get_similar_events(self, target_event, top_k=5, min_similarity=0.85):
        if target_event not in self.event_embeddings:
            return []
        
        target_emb = self.event_embeddings[target_event]
        
        similarities = []
        for event, emb in self.event_embeddings.items():
            if event != target_event:
                sim = cosine_similarity([target_emb], [emb])[0][0]
                if sim >= min_similarity:
                    similarities.append((event, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def calculate_baseline_prr(self, drug, event):
        a = self.combo_counts.get((drug, event), 0)
        
        if a == 0:
            return None
        
        total_d = self.drug_counts.get(drug, 0)
        b = total_d - a
        total_e = self.event_counts.get(event, 0)
        c = total_e - a
        d = self.total_reports - a - b - c
        
        if b <= 0 or c <= 0 or d <= 0:
            return None
        
        prr = (a * d) / (b * c)
        
        cont_table = np.array([[a, b], [c, d]])
        chi2, p_val, _, _ = stats.chi2_contingency(cont_table)
        
        is_signal = (prr >= 2) and (chi2 >= 4) and (a >= 3)
        
        return {
            'drug': drug,
            'event': event,
            'n_cases': a,
            'prr': prr,
            'chi2': chi2,
            'p_value': p_val,
            'is_signal': is_signal
        }
    
    def calculate_semantic_enhanced_prr(self, drug, event, top_k=5, min_similarity=0.85):
        baseline = self.calculate_baseline_prr(drug, event)
        
        if not baseline:
            return None
        
        similar_events = self.get_similar_events(event, top_k=top_k, min_similarity=min_similarity)
        
        if not similar_events:
            baseline['enhanced_prr'] = baseline['prr']
            baseline['similar_events_used'] = 0
            baseline['improvement'] = 0.0
            return baseline
        
        prr_scores = [baseline['prr']]
        weights = [1.0]
        
        similar_events_info = []
        
        for similar_event, similarity in similar_events:
            similar_result = self.calculate_baseline_prr(drug, similar_event)
            
            if similar_result and similar_result['n_cases'] >= 3:
                prr_scores.append(similar_result['prr'])
                weights.append(similarity * 0.5)
                
                similar_events_info.append({
                    'event': similar_event,
                    'similarity': similarity,
                    'prr': similar_result['prr'],
                    'n_cases': similar_result['n_cases']
                })
        
        if len(prr_scores) > 1:
            enhanced_prr = np.average(prr_scores, weights=weights)
        else:
            enhanced_prr = baseline['prr']
        
        baseline['enhanced_prr'] = enhanced_prr
        baseline['similar_events_used'] = len(similar_events_info)
        baseline['improvement'] = ((enhanced_prr - baseline['prr']) / baseline['prr']) * 100
        baseline['similar_events'] = similar_events_info
        
        is_enhanced_signal = (enhanced_prr >= 2) and (baseline['chi2'] >= 4) and (baseline['n_cases'] >= 3)
        baseline['is_enhanced_signal'] = is_enhanced_signal
        
        return baseline
    
    def compare_methods_on_validation(self, validation_cases):
        results = []
        
        for i, case in enumerate(validation_cases, 1):
            drug_pattern = case['drug']
            event = case['event'].upper()
            
            matching_drugs = [d for d in self.drug_counts.keys() 
                            if drug_pattern.upper() in d.upper()]
            
            if not matching_drugs:
                print(f"  {i}. {drug_pattern} - NO MATCHING DRUG")
                continue
            
            drug = max(matching_drugs, key=lambda d: self.drug_counts[d])
            
            result = self.calculate_semantic_enhanced_prr(drug, event, top_k=5, min_similarity=0.85)
            
            if result:
                result['expected_prr'] = case.get('expected_prr', 'unknown')
                result['description'] = case.get('description', '')
                results.append(result)
                
                
                if result['similar_events_used'] > 0:
                    print(f"Top similar events:")
                    for se in result['similar_events'][:3]:
                        print(f"{se['event'][:50]:50s} (sim={se['similarity']:.3f}, PRR={se['prr']:.2f})")
            else:
                print(f"  {i}. {drug} + {event} - NO DATA")
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    from validation_cases import validation_cases
    
    val_cases = validation_cases()
    pairs = pd.read_parquet(RESULTS_DIR / 'drug_event_pairs.parquet')
    semantic_prr = SemanticEnhancedPRR(
        pairs, 
        embeddings_path=EMBEDDINGS_FILE,
        unique_aes_path=UNIQUE_AES_FILE
    )
    
    comparison_results = semantic_prr.compare_methods_on_validation(val_cases)
    
    if len(comparison_results) > 0:
        output_path = RESULTS_DIR / 'semantic_enhanced_validation.csv'
        comparison_results.to_csv(output_path, index=False)
        avg_improvement = comparison_results['improvement'].mean()
        improved_count = (comparison_results['improvement'] > 0).sum()
        summary_cols = ['drug', 'event', 'prr', 'enhanced_prr', 'improvement', 'similar_events_used']
        print(comparison_results[summary_cols].to_string(index=False))
 