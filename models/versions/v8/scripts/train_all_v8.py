import os
import argparse
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--no-calibration', action='store_true')
    args = parser.parse_args()

    # Ensure project root is on sys.path for child processes too
    _CUR = os.path.dirname(__file__)
    _ROOT = os.path.abspath(os.path.join(_CUR, '..', '..', '..', '..'))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    symbols = [s.upper() for s in cfg.get('symbols', [])]
    print('Training symbols:', symbols)
    for sym in symbols:
        train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_v8.py'))
        cmd = [sys.executable, train_path, '--symbol', sym]
        if args.epochs is not None:
            cmd += ['--epochs', str(args.epochs)]
        if args.max_folds is not None:
            cmd += ['--max-folds', str(args.max_folds)]
        if args.no_calibration:
            cmd += ['--no-calibration']
        print('Running:', ' '.join(cmd))
        env = os.environ.copy()
        env['PYTHONPATH'] = _ROOT + os.pathsep + env.get('PYTHONPATH', '')
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print(f'Warning: training for {sym} exited with code {rc}')


if __name__ == '__main__':
    main()
import subprocess
import sys
import os
import yaml


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    symbols = cfg.get('symbols', [])
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, 'train_v8.py')
    for sym in symbols:
        print(f"\n==== Training {sym} ====")
        cmd = [sys.executable, script, '--symbol', sym]
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"Training failed for {sym} with code {ret}")
            break


if __name__ == '__main__':
    main()
