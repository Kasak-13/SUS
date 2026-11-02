import sys, os
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.preprocess import load_and_preprocess
from src.sus_detector import SUSDetector

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    X_stream_D = data_real['X_stream']
    
    print("\n=== Testing Scenario D with Safeguards AND Real Adaptation ===")
    detector = SUSDetector()
    detector.fit(X_train)
    
    decisions = []
    for x in X_stream_D:
        _, decision = detector.update_and_predict(x)
        decisions.append(decision)
        
    valid_decisions = [d for d in decisions if d is not None]
    counts = Counter(valid_decisions)
    total = len(valid_decisions)
    
    print("Decision Breakdown:")
    for k in ['KEEP', 'ADAPT', 'HOLD']:
        c = counts.get(k, 0)
        print(f"  {k}: {c} ({(c/total)*100:.1f}%)")

if __name__ == '__main__':
    main()
