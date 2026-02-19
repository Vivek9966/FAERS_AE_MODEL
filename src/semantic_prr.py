
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import faiss

RESULTS_DIR = Path("/home/vivekbisht/Desktop/faers-signal-detection/results_private")
EMBEDDINGS_FILE = RESULTS_DIR / 'ae_embeddings_sapbert.npy'
UNIQUE_AES_FILE = RESULTS_DIR / 'unique_aes.txt'

class SemanticEnhancedPRR:
    def __init__(self, pairs_df, embeddings_path=EMBEDDINGS_FILE, unique_aes_path=UNIQUE_AES_FILE):
        
        self.pairs = pairs_df
       
        pairs_df['event'] = pairs_df['event'].str.upper()
        pairs_df['drug'] = pairs_df['drug'].str.upper()


        MIN_EVENT_FREQ=50 
        eve_freq=pairs_df['event'].value_counts()
        eve_keep = eve_freq[eve_freq>=MIN_EVENT_FREQ].index
        pairs_df = pairs_df[pairs_df['event'].isin(eve_keep)].copy()  
        self.total_reports = len(pairs_df)      
        self.drug_counts = pairs_df.groupby('drug').size().to_dict()
        self.event_counts = pairs_df.groupby('event').size().to_dict()
        self.combo_counts = pairs_df.groupby(['drug', 'event']).size().to_dict()
       
        self.de_mat =(pairs_df.groupby(['event','drug'])).size().unstack(fill_value=0)

        # #-------------------------------log--------------------------------# To check the mem requirement tweak MIN_event FREQ and the nneighbotrs
        # nnz = np.count_nonzero(self.de_mat.values)
        # memory_ = self.de_mat.memory_usage(deep =True).sum()
        # print("After filtering MIN_EVENT_FREQ=20")
        # print(f"  Number of remaining events (rows in de_mat): {self.de_mat.shape[0]:,d}")
        # print(f"  Number of drugs (columns in de_mat):         {self.de_mat.shape[1]:,d}")
        # print(f"  Shape of de_mat:                             {self.de_mat.shape}")
        # print(f"  Non-zero elements (sparsity info):           {nnz:,d} / {self.de_mat.size:,d}  "
        #     f"({nnz/ self.de_mat.size:.3%} density)")
        # print(f"  Memory estimate for dense array:   ~{memory_ / 1e9:.1f} GB (float64)")
        # # X = normalize(self.de_mat.values, norm='l2')
        # # simi =cosine_similarity(X)
        # # self.co_occurence = pd.DataFrame(simi,index=self.de_mat.index,columns =self.de_mat.index)
        # ------------------------------log--------------------------------------
        X = normalize(self.de_mat.values, norm='l2').astype(np.float32)

        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        # nn= NearestNeighbors(
        #     n_neighbors=50 , metric='euclidean',algorithm='auto' , n_jobs=-1)
        
        # nn.fit(X)
        similarity ,ind =index.search(X,51)
        events = self.de_mat.index.tolist()
        self.cooc_knn = {}

        for i ,eve in enumerate(events):
            sim = similarity[i,1:51]
            ngbr_i = ind[i,1:51]
            ngbr = [(events[j],float(s)) for j,s  in zip(ngbr_i,sim) if j!=i]
            self.cooc_knn[eve] = ngbr[:50]

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
    
    def hybrid_simi(self,eve_i , eve_j ,alpha=.5):
        sem = 0.0
        if eve_i in self.event_embeddings and eve_j in self.event_embeddings:
            sem = cosine_similarity(
                [self.event_embeddings[eve_i]],
                [self.event_embeddings[eve_j]]
            )[0][0]

        cooc = 0.0
        for ev, sim in self.cooc_knn.get(eve_i, []):
            if ev == eve_j:
                cooc = sim
                break

        return alpha * sem + (1 - alpha) * cooc
    def hybrid_neighbours(self, tar_eve, min_simi=0.3, top_k=5, alpha=0.5):
        scores = []

        for e, _ in self.cooc_knn.get(tar_eve, []):
            sim = self.hybrid_simi(tar_eve, e, alpha)
            if sim >= min_simi:
                scores.append((e, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    def pooled_prr(self,drg,eve,top_k=5,alpha=.5):

        a=self.combo_counts.get((drg,eve),0)
        if a ==0:
            return None
        ngbr=self.hybrid_neighbours(eve,top_k=top_k,alpha=alpha)

        pooled_a = float(a)
        pooled_eve = []
        for ev,sm in ngbr:
            cnt = self.combo_counts.get((drg,ev),0)
            if cnt >= 1:
                pooled_a += sm*cnt
                pooled_eve.append({
                    'event':ev , 'similarity':sm ,'n_classes':cnt 
                })
        total_d = self.drug_counts.get(drg,0)
        total_e = self.event_counts.get(eve, 0)
        b = total_d - pooled_a
        c = total_e - pooled_a
        d = self.total_reports -pooled_a-b-c

        if min(b,c,d)<=0:
            return None
        prr =( pooled_a*d) / (b*c)
        cont_table = np.array([[pooled_a,b],[c,d]])
        chi2,p_val,_,_ = stats.chi2_contingency(cont_table)
        return {
        'drug': drg,
        'event': eve,
        'pooled_cases': pooled_a,
        'prr': prr,
        'chi2': chi2,
        'p_value': p_val,
        'neighbors_used': len(pooled_eve),
        'neighbors': pooled_eve,
        'is_signal': (prr >= 2 and chi2 >= 4 and pooled_a >= 3)
    }
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
            
            # result = self.pooled_prr(drug,event,top_k=5,alpha=0.5)
            
            # if result:
            #     result['expected_prr'] = case.get('expected_prr', 'unknown')
            #     result['description'] = case.get('description', '')
            #     results.append(result)
                
            baseline = self.calculate_baseline_prr(drug,event)
            result = self.pooled_prr(drug , event ,top_k=5 ,alpha=.5)
            if result:
                if baseline:
                    result['baseline_prr']  =baseline['prr']
                    result['baseline_n_cases'] = baseline['n_cases']
                else:
                    result['baseline_prr']=np.nan
                    result['baseline_n_cases'] = 0
                result['expected_prr'] = case.get('expected_prr','unknown')
                result['description'] = case.get('description','')
                results.append(result)
                if result['neighbors_used'] > 0:
                    print("Top hybrid neighbors:")
                    for se in result['neighbors'][:3]:
                        print(f"{se['event'][:50]:50s} (sim={se['similarity']:.3f})")

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
        # avg_improvement = comparison_results['improvement'].mean()
        # improved_count = (comparison_results['improvement'] > 0).sum()
        # summary_cols = ['drug', 'event', 'prr', 'enhanced_prr', 'improvement', 'similar_events_used']
        summary_cols = ['drug', 'event', 'prr', 'pooled_cases', 'neighbors_used']
        print(comparison_results[summary_cols].to_string(index=False))
 