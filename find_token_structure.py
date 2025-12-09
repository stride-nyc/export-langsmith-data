"""Quick script to find usage_metadata in trace structure."""
import json

# Load existing backup data
with open(r'C:\Users\Ken Judy\source\repos\neotalogic-ai-demo-214120cfb40b\Assessment\data\neota_agent_traces_1000 copy.json', encoding='utf-8') as f:
    data = json.load(f)

traces = data.get('traces', [])
print(f"Total traces: {len(traces)}")

# Search for usage_metadata in nested structure
def find_usage_metadata(obj, path='', depth=0):
    if depth > 10:  # Limit recursion
        return []

    results = []
    if isinstance(obj, dict):
        if 'usage_metadata' in obj:
            results.append((path, obj['usage_metadata']))
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            results.extend(find_usage_metadata(v, new_path, depth+1))
    elif isinstance(obj, list) and depth < 5:  # Limit list traversal
        for i, item in enumerate(obj[:3]):  # Only check first 3 items
            results.extend(find_usage_metadata(item, f"{path}[{i}]", depth+1))
    return results

# Search first 20 traces
results = find_usage_metadata(traces[:20])
print(f"\nFound {len(results)} usage_metadata instances in first 20 traces")

if results:
    print(f"\nExample location: {results[0][0]}")
    print(f"Example data:\n{json.dumps(results[0][1], indent=2)}")
else:
    print("\nNo usage_metadata found. Checking what fields DO exist...")
    sample = traces[0] if traces else {}
    print(f"Root trace keys: {list(sample.keys())}")
    if 'child_runs' in sample and sample['child_runs']:
        child = sample['child_runs'][0]
        print(f"Child run keys: {list(child.keys())}")
        if 'outputs' in child:
            print(f"Child outputs keys: {list(child.get('outputs', {}).keys())}")
