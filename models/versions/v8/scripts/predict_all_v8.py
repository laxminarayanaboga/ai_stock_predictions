import os
import sys
import subprocess
import yaml


def main():
    # Ensure project root for children
    _CUR = os.path.dirname(__file__)
    _ROOT = os.path.abspath(os.path.join(_CUR, '..', '..', '..', '..'))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    symbols = [s.upper() for s in cfg.get('symbols', [])]
    print('Predicting for symbols:', symbols)
    for sym in symbols:
        predict_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'predict_v8.py'))
        cmd = [sys.executable, predict_path, '--symbol', sym]
        env = os.environ.copy()
        env['PYTHONPATH'] = _ROOT + os.pathsep + env.get('PYTHONPATH', '')
        print('Running:', ' '.join(cmd))
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print(f'Warning: prediction for {sym} exited with code {rc}')


if __name__ == '__main__':
    main()
