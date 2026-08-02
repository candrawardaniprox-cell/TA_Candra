import os

def clean_future_imports(directory):
    for root, dirs, files in os.walk(directory):
        if 'venv' in dirs:
            dirs.remove('venv')
        if '.env' in dirs:
            dirs.remove('.env')
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except Exception:
                    continue
                
                # Check if we have multiple from __future__ import annotations
                
                if len(future_lines) > 1:
                    print(f"Fixing multiple future imports in {filepath}")
                    # Remove all but the first one, actually if line 0 is the one we added, let's remove line 0
                    if future_lines[0] == 0:
                        lines.pop(0)
                    else:
                        # just remove duplicates after the first
                        for i in reversed(future_lines[1:]):
                            lines.pop(i)
                            
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

if __name__ == '__main__':
    clean_future_imports('.')
