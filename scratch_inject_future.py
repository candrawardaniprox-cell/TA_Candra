import os

def insert_future(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return
        
    if 'from __future__ import annotations' in content:
        return
        
    lines = content.split('\n')
    insert_idx = 0
    
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith('"""'):
            if first_line.count('"""') >= 2 and len(first_line) >= 6 and first_line.endswith('"""'):
                insert_idx = 1
            else:
                for i in range(1, len(lines)):
                    if '"""' in lines[i]:
                        insert_idx = i + 1
                        break
        elif first_line.startswith("'''"):
            if first_line.count("'''") >= 2 and len(first_line) >= 6 and first_line.endswith("'''"):
                insert_idx = 1
            else:
                for i in range(1, len(lines)):
                    if "'''" in lines[i]:
                        insert_idx = i + 1
                        break
    
    lines.insert(insert_idx, 'from __future__ import annotations')
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Added to {filepath}")
    except Exception as e:
        print(f"Failed to write {filepath}: {e}")

for root, dirs, files in os.walk('.'):
    if 'venv' in dirs: dirs.remove('venv')
    if '.env' in dirs: dirs.remove('.env')
    for file in files:
        if file.endswith('.py'):
            insert_future(os.path.join(root, file))
