"""Restore omitted inputs from an existing local archive, verifying all hashes first."""
from pathlib import Path
import argparse, hashlib, json, shutil

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='Original experiments directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    archive = args.archive.resolve()
    target_root = (root / 'experiments').resolve()
    rows = json.loads((root / 'third_party_inputs.json').read_text(encoding='utf-8'))['files']
    pending = []
    for row in rows:
        src = (archive / row['path']).resolve()
        dst = (target_root / row['path']).resolve()
        if not src.is_relative_to(archive) or not dst.is_relative_to(target_root):
            raise ValueError('Input path escapes the selected directory')
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Missing or changed archived input: ' + row['path'])
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Refusing to overwrite a different input: ' + row['path'])
        pending.append((src, dst))
    for src, dst in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src != dst:
            shutil.copy2(src, dst)
    print('Restored and verified', len(pending), 'third-party inputs; these paths are git-ignored.')

if __name__ == '__main__':
    main()
