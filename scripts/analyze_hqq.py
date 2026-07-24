import json
with open('runs/phase3/agent_a_backend/hqq_results.jsonl') as f:
    for line in f:
        d = json.loads(line)
        print(f"{d['model']} {d['condition']} {d['type']} {d['prompt_idx']}: valid={d.get('valid','-')} correct={d.get('correct','-')} fdp={d.get('first_divergence_token','-')} byte_id={d.get('byte_identical','-')}")
