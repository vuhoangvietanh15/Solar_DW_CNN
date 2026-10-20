"""Masked self-supervision: measured values are targets, never called clean."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import tf, configure
from .model import DW_CNN, masked_mse
from .data import mask_values, normalize


class MaskedSequence(tf.keras.utils.Sequence):
    def __init__(self, x, batch_size, period, seed, training):
        self.x, self.batch_size, self.period = x, batch_size, period
        self.rng = np.random.default_rng(seed)
        self.training = training
        self.indices = np.arange(len(x))
        self.phases = self.rng.integers(period, size=len(x))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return mask_values(self.x[ids], self.phases[ids], self.period)

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.indices)
            self.phases = self.rng.integers(self.period, size=len(self.x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/prepared'))
    p.add_argument('--run', type=Path, default=Path('runs/solar_masked'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--filters', type=int, default=64)
    p.add_argument('--mask-period', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--steps-per-epoch', type=int, help='Smoke check only; omit for training')
    p.add_argument('--validation-steps', type=int, help='Smoke check only; omit for training')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.filters, args.patience) < 1 or args.mask_period < 3 or args.learning_rate <= 0:
        p.error('Positive settings required; mask-period >=3')
    if args.run.exists():
        raise FileExistsError(f'Run already exists; choose --run with a new name: {args.run}')
    devices = configure(args.allow_cpu, args.mixed_precision, args.seed)
    meta = json.loads((args.data/'metadata.json').read_text())
    if args.mask_period > meta['length']:
        p.error('mask-period cannot exceed input length')
    with np.load(args.data/'train.npz') as z:
        train = normalize(z['values_db'], meta)
    with np.load(args.data/'val.npz') as z:
        val = normalize(z['values_db'], meta)
    tr = MaskedSequence(train, args.batch_size, args.mask_period, args.seed, True)
    va = MaskedSequence(val, args.batch_size, args.mask_period, args.seed+1, False)
    model = DW_CNN(meta['length'], args.filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1.), loss=masked_mse)
    args.run.mkdir(parents=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'data_metadata': meta, 'parameters': model.count_params(),
                   'tensorflow': tf.__version__, 'gpu': [d.name for d in devices],
                   'method': 'masked_self_supervision', 'clean_ground_truth': False,
                   'smoke_only': bool(args.steps_per_epoch or args.validation_steps)})
    (args.run/'config.json').write_text(json.dumps(config, indent=2))
    model.summary(print_fn=lambda s: print(s, flush=True))
    # Baseline on the same fixed held-out validation bins; no competing networks.
    total, count = 0., 0.
    for i in range(min(len(va), args.validation_steps or len(va))):
        x, y = va[i]
        total += float(np.sum((x[..., 0]-y[..., 0])**2 * y[..., 1]))
        count += float(y[..., 1].sum())
    baseline = total/count
    (args.run/'validation_baseline.json').write_text(json.dumps({'neighbor_interpolation_masked_mse_normalized': baseline}, indent=2))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(args.run/'best.keras'), monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=max(2, args.patience//3), min_lr=1e-7),
                 tf.keras.callbacks.CSVLogger(str(args.run/'history.csv')),
                 tf.keras.callbacks.TensorBoard(log_dir=str(args.run/'tensorboard')),
                 tf.keras.callbacks.TerminateOnNaN()]
    history = model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=callbacks,
                        steps_per_epoch=args.steps_per_epoch, validation_steps=args.validation_steps,
                        workers=1, use_multiprocessing=False, verbose=2)
    losses = np.asarray(history.history['val_loss'])
    if not np.isfinite(losses).all() or not (args.run/'best.keras').exists():
        raise RuntimeError('Nonfinite training or missing best checkpoint; inspect logs')
    best = float(losses.min())
    report = {'best_val_masked_mse_normalized': best,
              'neighbor_baseline_masked_mse_normalized': baseline,
              'beats_neighbor_baseline': bool(best < baseline),
              'note': 'Masked prediction risk, NOT clean-target error or scientific SNR.'}
    (args.run/'training_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Best model:', args.run/'best.keras')


if __name__ == '__main__':
    main()
