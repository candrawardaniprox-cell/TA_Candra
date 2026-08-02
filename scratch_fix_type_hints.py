import os

def fix_type_hints(directory):
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
                        content = f.read()
                except Exception as e:
                    print(f"Could not read {filepath}: {e}")
                    continue
                
                if not content.startswith('from __future__') and ('| None' in content or 'list[' in content or 'dict[' in content or 'tuple[' in content or 'set[' in content or 'bool |' in content or 'str |' in content or 'Path |' in content):
                    print(f"Fixing {filepath}")
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write('from __future__ import annotations\n' + content)
                    except Exception as e:
                        print(f"Could not write {filepath}: {e}")

if __name__ == '__main__':
    fix_type_hints('.')
