"""Measure response to injected spectral features; not a clean-target benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import configure, tf
from . import model as model_module
from .infer import blind_predict
from .data import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/prepared'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    if a.samples < 1:
        p.error('samples must be positive')
    if a.output.exists():
        raise FileExistsError(f'Output exists: {a.output}')
    cfg = json.loads((a.run/'config.json').read_text())
    meta = cfg['data_metadata']
    if json.loads((a.data/'metadata.json').read_text()) != meta:
        raise ValueError('Checkpoint/data metadata mismatch')
    configure(a.allow_cpu, cfg['mixed_precision'], cfg['seed'])
    net = tf.keras.models.load_model(a.run/'best.keras', compile=False)
    with np.load(a.data/'test.npz') as z:
        ids = np.linspace(0, len(z['values_db'])-1, min(a.samples, len(z['values_db']))).astype(int)
        x = normalize(z['values_db'][ids], meta)
    base = blind_predict(net, x, cfg['mask_period'])
    bins = np.arange(x.shape[1])
    # Evaluate multiple locations, avoiding the spectrum boundary.
    centers = np.linspace(x.shape[1]//4, 3*x.shape[1]//4, len(x)).astype(int)
    rows, curves = [], []
    for sigma in [.5, 1., 2., 4., 8., 16.]:
        injection = np.exp(-.5*((bins[None, :]-centers[:, None])/sigma)**2).astype(np.float32)
        # Amplitude = one training-set std in recorded dB units.
        response = blind_predict(net, x+injection, cfg['mask_period'])-base
        at_peak = response[np.arange(len(x)), centers]
        row = {'sigma_bins': sigma, 'peak_response_median': float(np.median(at_peak)),
               'integrated_response_median': float(np.median(response.sum(axis=1)/injection.sum(axis=1))),
               'response_rmse_normalized': float(np.sqrt(np.mean((response-injection)**2)))}
        rows.append(row)
        curves.append((injection[0], response[0]))
    # Adjacent-bin correlations of temporal DIFFERENCES are a heuristic only.
    # Shared sky variation/RFI also contributes, so this is not a noise estimate.
    diffs = np.diff(x, axis=0)
    corr = None
    if diffs.size and diffs[:, :-1].std() > 0 and diffs[:, 1:].std() > 0:
        corr = float(np.corrcoef(diffs[:, :-1].ravel(), diffs[:, 1:].ravel())[0, 1])
    report = {'injections': rows, 'sampled_frame_indices': ids.tolist(),
              'injection_amplitude_db': meta['normalization']['scale_db'],
              'adjacent_bin_correlation_of_sampled_temporal_differences': corr,
              'interpretation': 'Response 1 means retention, 0 means suppression. Narrow astronomical lines and narrow RFI are indistinguishable without labels. This is a sensitivity probe, NOT clean-target accuracy. Correlation is descriptive, not proof of noise independence.'}
    a.output.mkdir(parents=True)
    (a.output/'injection_response.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, row, (inj, response) in zip(axes.ravel(), rows, curves):
        ax.plot(bins, inj, label='Known added feature')
        ax.plot(bins, response, label='Change in model output')
        ax.set_xlim(centers[0]-50, centers[0]+50)
        ax.set_title(f"sigma={row['sigma_bins']} bins")
        ax.set_xlabel('Recorded bin index')
        ax.legend(fontsize=7)
    fig.suptitle('Feature-retention probe; arbitrary log-unit injection, no clean reference')
    fig.savefig(a.output/'injection_response.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
