import urllib.request, json
req = urllib.request.urlopen('http://127.0.0.1:5000/api/scenario/A')
data = json.loads(req.read().decode('utf-8'))
adapt_steps = [d for d in data if d['decision'] == 'ADAPT']
print(f"Total ADAPT in Scenario A: {len(adapt_steps)}")
for d in adapt_steps[:10]:
    print(f"Step {d['step']}: decision={d['decision']}, drift_score={d['drift_score']:.4f}, anom_frac={d['anomaly_fraction']:.4f}")

