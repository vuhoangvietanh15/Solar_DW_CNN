"""Blind prediction at every bin, using the exact training-time masks."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .runtime import tf, configure
from . import model as model_module  # Registers serializable classes before load_model.
from .data import normalize, mask_values, read_day


def blind_predict(model, x, period=8, batch_size=16):
    result = np.empty_like(x)
    predict_fn = tf.function(
        lambda values: model(values, training=False),
        input_signature=[tf.TensorSpec([None, x.shape[1], 2], tf.float32)])
    for start in range(0, len(x), batch_size):
        b = x[start:start+batch_size]
        for phase in range(period):
            inputs, target = mask_values(b, np.full(len(b), phase), period)
            pred = np.asarray(predict_fn(inputs))[..., 0]
            selection = target[..., 1].astype(bool)
            result[start:start+len(b)][selection] = pred[selection]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite inference output')
    return result


def plot_results(raw, filtered, f, times, out, day):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone, timedelta
    from matplotlib.colors import TwoSlopeNorm
    tz = timezone(timedelta(hours=7))
    labels = [datetime.fromtimestamp(t, tz).strftime('%H:%M:%S') for t in times]
    lo, hi = np.percentile(raw, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    residual = raw-filtered
    vmax = max(float(np.percentile(np.abs(residual), 99)), 1e-6)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    for ax, image, title in zip(axes, [raw, filtered, residual],
                                ['Recorded input', 'DW-CNN blind prediction', 'Input - prediction']):
        if ax is axes[2]:
            im = ax.imshow(image.T, origin='lower', aspect='auto', interpolation='nearest',
                           cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        else:
            im = ax.imshow(image.T, origin='lower', aspect='auto', interpolation='nearest', cmap='inferno', vmin=lo, vmax=hi)
        ticks = np.linspace(0, len(raw)-1, min(8, len(raw))).astype(int)
        ax.set_xticks(ticks, [labels[t] for t in ticks], rotation=20)
        fticks = np.linspace(0, len(f)-1, 6).astype(int)
        ax.set_yticks(fticks, [f'{f[t]:.4f}' for t in fticks])
        ax.set_ylabel('Recorded MHz (unverified)')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Recorded dB' if ax is not axes[2] else 'dB difference')
    axes[-1].set_xlabel('Time UTC+07 (each column is one observation; gaps not to scale)')
    fig.suptitle(f'{day} — no clean ground truth; frequency provenance unverified')
    fig.savefig(out/'waterfall.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for i in sorted(set([0, len(raw)//2, len(raw)-1])):
        axes[0].plot(f, raw[i], alpha=.45, label=f'Input {labels[i]}')
        axes[0].plot(f, filtered[i], linewidth=1, label=f'Prediction {labels[i]}')
    axes[0].set(xlabel='Recorded frequency MHz (unverified)', ylabel='Recorded dB')
    axes[0].legend(ncol=2)
    # Input is already an average in log-domain. Do NOT call this absolute flux.
    axes[1].plot(times-times[0], raw.mean(axis=1), label='Mean recorded dB')
    axes[1].plot(times-times[0], filtered.mean(axis=1), label='Mean predicted dB')
    axes[1].set(xlabel='Elapsed seconds', ylabel='Mean of recorded log values')
    axes[1].legend()
    fig.savefig(out/'spectra_and_trend.png', dpi=180)
    plt.close(fig)


def diagnostic_report(raw, filtered, scale):
    residual = raw-filtered
    return {'frames': len(raw), 'bins': raw.shape[1],
            'rms_change_db': float(np.sqrt(np.mean(residual**2))),
            'mean_shift_db': float(np.mean(filtered-raw)),
            'max_abs_change_db': float(np.max(np.abs(residual))),
            'mean_abs_frame_mean_shift_db': float(np.mean(np.abs(residual.mean(axis=1)))),
            'masked_prediction_mse_normalized': float(np.mean((residual/scale)**2)),
            'clean_reference_available': False,
            'note': 'Changes and masked risk only. No claim of RFI removal, SNR gain, flux or phase preservation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/prepared'))
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--day', help='Predict a raw observing-day folder instead of a prepared split')
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Choose a new output directory: {a.output}')
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    cfg = json.loads((a.run/'config.json').read_text())
    meta = cfg['data_metadata']
    configure(a.allow_cpu, cfg['mixed_precision'], cfg['seed'])
    net = tf.keras.models.load_model(a.run/'best.keras', compile=False)
    if a.day:
        b = read_day(a.source or Path(meta['source']), a.day)
        raw, f, times, files = b['values'], b['frequency'], b['times'], b['files']
    else:
        data_meta = json.loads((a.data/'metadata.json').read_text())
        if data_meta != meta:
            raise ValueError('Prepared data metadata does not match checkpoint')
        with np.load(a.data/f'{a.split}.npz') as b:
            raw, f, times, files = b['values_db'], b['frequency_mhz'], b['timestamps'], b['files']
    # Check exact saved training grid even for raw-day inference.
    with np.load(Path(cfg['data'])/'train.npz') as b:
        reference = b['frequency_mhz']
    if f.shape != reference.shape or not np.allclose(f, reference, rtol=0, atol=1e-6):
        raise ValueError('Different observing setup/frequency grid. Prepare and train a separate run.')
    x = normalize(raw, meta)
    prediction = blind_predict(net, x, cfg['mask_period'], a.batch_size)
    n = meta['normalization']
    filtered = prediction*n['scale_db'] + n['center_db']
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'prediction.npz', input_db=raw, prediction_db=filtered,
                        removed_db=raw-filtered, frequency_mhz=f, timestamps=times, files=files)
    report = diagnostic_report(raw, filtered, n['scale_db'])
    # Baseline uses the same blind mask procedure as the model.
    baseline = np.empty_like(x)
    for phase in range(cfg['mask_period']):
        bx, by = mask_values(x, np.full(len(x), phase), cfg['mask_period'])
        sel = by[..., 1].astype(bool)
        baseline[sel] = bx[..., 0][sel]
    report['neighbor_masked_mse_normalized'] = float(np.mean((baseline-x)**2))
    (a.output/'report.json').write_text(json.dumps(report, indent=2))
    (a.output/'provenance.json').write_text(json.dumps({'run': str(a.run.resolve()), 'config': cfg}, indent=2))
    with (a.output/'frames.csv').open('w') as handle:
        w = csv.writer(handle)
        w.writerow(['file', 'timestamp_utc', 'mean_input_db', 'mean_prediction_db', 'rms_change_db'])
        for i in range(len(raw)):
            w.writerow([files[i], times[i], float(raw[i].mean()), float(filtered[i].mean()),
                        float(np.sqrt(np.mean((raw[i]-filtered[i])**2)))])
    # Never combine different days into one fabricated time-continuous waterfall.
    day_keys = np.array(['_'.join(Path(p).name.split('_')[:3]) for p in files])
    for day in np.unique(day_keys):
        ids = np.where(day_keys == day)[0]
        dest = a.output/day
        dest.mkdir()
        plot_results(raw[ids], filtered[ids], f, times[ids], dest, day)
    print(json.dumps(report, indent=2))
    print('Outputs:', a.output.resolve())


if __name__ == '__main__':
    main()
