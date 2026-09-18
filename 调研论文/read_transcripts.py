import json

def get_last_model_response(file_path):
    responses = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('source') == 'MODEL' and data.get('content'):
                    responses.append(data.get('content'))
        return responses[-1] if responses else 'No response found'
    except Exception as e:
        return str(e)

path1 = r'C:\Users\李佳昱\.gemini\antigravity\brain\2f87000d-8faf-4d0d-81d3-8cea3c7df260\.system_generated\logs\transcript.jsonl'
path2 = r'C:\Users\李佳昱\.gemini\antigravity\brain\10521874-24fc-40d9-a9a9-6bb130848697\.system_generated\logs\transcript.jsonl'

with open('summary_out.md', 'w', encoding='utf-8') as f:
    f.write('--- GitHub Projects Summary ---\n')
    f.write(get_last_model_response(path1))
    f.write('\n\n--- Research Papers Summary ---\n')
    f.write(get_last_model_response(path2))
