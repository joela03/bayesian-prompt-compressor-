import json
import numpy as np

with open('data/results/benchmark/our_results.json') as f:
    ours = json.load(f)
with open('data/results/benchmark/llmlingua_results.json') as f:
    lingua = json.load(f)

our_times = [v['time_seconds'] for v in ours.values() if 'time_seconds' in v]
llm_times = [v['time_seconds'] for v in lingua.values() 
             if 'time_seconds' in v and not v.get('error')]

print(f"Ours:      avg {np.mean(our_times):.2f}s  median {np.median(our_times):.2f}s")
print(f"LLMLingua: avg {np.mean(llm_times):.2f}s  median {np.median(llm_times):.2f}s")