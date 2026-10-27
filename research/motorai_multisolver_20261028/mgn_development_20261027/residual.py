"""Equal63-label log-residual regression, fit-only preprocessing and group CV."""
import numpy as np

SCALE = {'Ref': 1., 'SC': 2., 'HalfSC': 1.5}


def feature_matrix(rows, embeddings=None, source_keys=None):
    base = np.array([[r['current']/SCALE[r['model']]/460, r['beta']/90,
                      np.log(r['speed']/16000), np.log(SCALE[r['model']])] for r in rows])
    if embeddings is None:
        return base
    lookup = {str(key): i for i, key in enumerate(source_keys)}
    return np.column_stack([base, embeddings[[lookup[r['source_key']] for r in rows]]])


def scaling(x):
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1
    sd[:4] *= 2
    if x.shape[1] > 4:
        sd[4:] *= np.sqrt(x.shape[1]-4)
    return mean, sd


def distances(x, y):
    return np.maximum((x*x).sum(1)[:, None]+(y*y).sum(1)[None]-2*x@y.T, 0)


def fit(x, target, multiplier, regularization):
    mean, sd = scaling(x)
    z = (x-mean)/sd
    d2 = distances(z, z)
    d = np.sqrt(d2[np.triu_indices(len(z), 1)])
    bandwidth = max(float(np.median(d[d > 1e-10])), 1e-8)*multiplier
    kernel = np.exp(-d2/(2*bandwidth**2))
    center = float(target.mean())
    alpha = np.linalg.solve(kernel+regularization*np.eye(len(z)), target-center)
    return dict(mean=mean, sd=sd, z=z, bandwidth=bandwidth, center=center, alpha=alpha,
                multiplier=multiplier, regularization=regularization)


def predict(model, x):
    z = (x-model['mean'])/model['sd']
    return model['center']+np.exp(-distances(z, model['z'])/(2*model['bandwidth']**2))@model['alpha']


def select_and_fit(x, rows):
    truth = np.array([r['truth'] for r in rows])
    hybrid = np.array([r['hybrid'] for r in rows])
    assert len(rows) == 63 and np.all(truth > 0) and np.all(hybrid > 0)
    target = np.log(truth/hybrid)
    groups = np.array([r['source_key'] for r in rows])
    scores = []
    for multiplier in [.5, 1., 2.]:
        for regularization in [1e-6, .001, .1]:
            predictions = np.full(len(rows), np.nan)
            for group in np.unique(groups):
                held = groups == group
                model = fit(x[~held], target[~held], multiplier, regularization)
                predictions[held] = hybrid[held]*np.exp(predict(model, x[held]))
            assert np.isfinite(predictions).all()
            scores.append(dict(multiplier=multiplier, regularization=regularization,
                               calibration_group_CV_wMAE_percent=float(100*np.abs(predictions-truth).sum()/truth.sum())))
    best = min(scores, key=lambda s:s['calibration_group_CV_wMAE_percent'])
    return fit(x, target, best['multiplier'], best['regularization']), scores
