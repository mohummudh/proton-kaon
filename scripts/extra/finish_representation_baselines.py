"""Drain completed baseline representations while GPU jobs run; then render figures."""
import time
from representation_baselines import OUT
from evaluate_representation_baselines import evaluate
from plot_representation_baselines import main as plot


def main():
    expected = {f'{m}{n}_s{s}' for m in ['random', 'ae', 'vae', 'masked_ae']
                for n in ['', '_n100', '_n1000'] for s in [0, 1, 2]}
    expected |= {m + n for m in ['pixel_pca', 'endpoint', 'endpoint_pca', 'fulltrack', 'fulltrack_pca']
                 for n in ['', '_n100', '_n1000']}
    done = set(); start = time.time()
    while done != expected:
        for stem in sorted(expected - done):
            p = OUT / 'representations' / (stem + '.npy')
            if not p.exists() or time.time() - p.stat().st_mtime < 2: continue
            stages = ['probes'] if '_n100' in stem else ['probes', 'cluster', 'mass_prediction']
            evaluate(p, stages); done.add(stem)
            print(f'FINISHED {len(done)}/{len(expected)} {stem}', flush=True)
        if time.time() - start > 14400: raise RuntimeError('Training/evaluation did not finish within four hours')
        if done != expected: time.sleep(10)
    plot()
    (OUT / 'COMPLETE.txt').write_text('All 51 representations evaluated; paper figures rendered.\n')


if __name__ == '__main__': main()
