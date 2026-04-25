import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

def principalComponent(train, var=None, comp=None):
    mu = np.mean(train, axis=1, keepdims=True)
    X = train-mu
    S = np.dot(X, X.T)/train.shape[1]
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    if comp is not None:
        k = comp
    elif var is not None:
        tot,s,k = np.sum(vals), 0, 0
        for i,v in enumerate(vals):
            s+=v
            if (s/tot >= var):
                k = i + 1
                break
    else:
        k = train.shape[0]
    Up = vecs[:, :k]
    Y = np.dot(Up.T, X)
    return Y, Up, vals[:k], mu


def meanSquareErr(yt, yp):
    return np.mean((yt-yp)**2)


def load_idx_images(fname):
    with gzip.open(fname, 'rb') as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Image file format is wrong")
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
    return imgs


def load_idx_labels(fname):
    with gzip.open(fname, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Label file format is wrong")
        lbls = np.frombuffer(f.read(), dtype=np.uint8)
    return lbls


def load_fashion_mnist():
    Xtr = load_idx_images('train-images-idx3-ubyte.gz')
    ytr = load_idx_labels('train-labels-idx1-ubyte.gz')
    Xte = load_idx_images('t10k-images-idx3-ubyte.gz')
    yte = load_idx_labels('t10k-labels-idx1-ubyte.gz')
    return Xtr, ytr, Xte, yte


def filter012(x, y):
    m = (y == 0) | (y == 1) | (y == 2)
    return x[m], y[m]


def preprocess_data(Xtr, Xte):
    Xtr = Xtr.reshape(Xtr.shape[0], -1) / 255.0
    Xte = Xte.reshape(Xte.shape[0], -1) / 255.0
    Ytr, Up, eigs, mu = principalComponent(Xtr.T, comp=10)
    Yte = np.dot(Up.T, Xte.T - mu)
    return Ytr.T, Yte.T, eigs


def compute_ssr(yl, yr):
    if (len(yl) == 0 or len(yr) == 0):
        return np.inf, None, None
    lm, rm = np.mean(yl), np.mean(yr)
    ssr = np.sum((yl - lm)**2) + np.sum((yr - rm)**2)
    return ssr, lm, rm


def train_stump(X, y):
    bf, bt, bssr, blv, brv = None, None, np.inf, None, None
    for fi in range(X.shape[1]):
        xs = X[:, fi]
        si = np.argsort(xs)
        xs, ys = xs[si], y[si]
        for i in range(len(xs)-1):
            if xs[i] == xs[i+1]:
                continue
            thr = (xs[i]+xs[i + 1])/2.0
            ssr, lv, rv = compute_ssr(ys[:i+1], ys[i+1:])
            if (ssr < bssr):
                bssr, bf, bt, blv, brv = ssr, fi, thr, lv, rv
    return {'feature': bf, 'threshold': bt, 'left_value': blv, 'right_value': brv, 'ssr': bssr}


def predict_stump(mdl, X):
    preds = np.zeros(X.shape[0])
    lm = X[:, mdl['feature']] <= mdl['threshold']
    preds[lm] = mdl['left_value']
    preds[~lm] = mdl['right_value']
    return preds


def samplesBootStrap(X, y, n, rng):
    ns = X.shape[0]
    out = []
    for _ in range(n):
        si = rng.integers(0, ns, size=ns)
        oob = np.where(~np.isin(np.arange(ns), si))[0]
        out.append((si, oob))
    return out


def bagging_predict(models, X):
    preds = np.array([predict_stump(m, X) for m in models])
    return np.mean(preds, axis=0)


def main():
    Xtr, ytr, Xte, yte = load_fashion_mnist()
    Xtr, ytr = filter012(Xtr, ytr)
    Xte, yte = filter012(Xte, yte)
    Xtr, Xte, eigs = preprocess_data(Xtr, Xte)

    print("Train shape after filtering:", Xtr.shape)
    print("Test shape after filtering:", Xte.shape)
    print("Train/Test shape after PCA:", Xtr.shape, Xte.shape)
    print("Top 10 eigenvalues:\n", eigs, "\n\n")

    sm = train_stump(Xtr, ytr.astype(float))
    sp = predict_stump(sm, Xte)
    smse = meanSquareErr(yte.astype(float), sp)

    print("Single regression decision stump")
    print("Best feature index:", sm['feature'])
    print("Best threshold:", sm['threshold'])
    print("Left region prediction:", sm['left_value'])
    print("Right region prediction:", sm['right_value'])
    print("Minimum SSR on train set:", sm['ssr'])
    print("Test MSE:", smse)

    rng = np.random.default_rng(2023115)
    boots = samplesBootStrap(Xtr, ytr.astype(float), 5, rng)
    bag_mdls, oob_errs = [], []

    for i, (bi, oi) in enumerate(boots):
        mdl = train_stump(Xtr[bi], ytr[bi].astype(float))
        bag_mdls.append(mdl)
        if (len(oi)>0):
            err = meanSquareErr(ytr[oi].astype(float), predict_stump(mdl, Xtr[oi]))
            oob_errs.append(err)
            print(f"Model {i+1} OOB MSE:", err)
        else:
            print(f"Model {i+1} OOB MSE: no OOB points found")

    bp = bagging_predict(bag_mdls, Xte)

    print("Bagging with 5 regression decision stumps")
    print("Average OOB MSE:", np.mean(oob_errs))
    print("Bagging test MSE:", meanSquareErr(yte.astype(float), bp))


    sp_up = np.round(np.clip(sp, 0, 2)).astype(int)
    bp_up = np.round(np.clip(bp, 0, 2)).astype(int)

    sort_idx = np.argsort(yte)
    y_true_s = yte[sort_idx]
    y_stump_s = sp_up[sort_idx]
    y_bag_s = bp_up[sort_idx]
    x_index = np.arange(len(yte))

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Q3 - Regression Stump vs Bagging on Fashion-MNIST', fontweight='bold', fontsize=14)

    ax1.plot(x_index, y_true_s, color='#1B1B1B', linewidth=2.5,
             label='True labels (0, 1, 2)', zorder=3)
    ax1.scatter(x_index, y_stump_s, color='#0081A7', s=20, alpha=0.65,
                label='Single Stump', zorder=2)
    ax1.scatter(x_index, y_bag_s, color='#F45D01', s=20, alpha=0.65,
                label='Bagged (5 stumps)', zorder=1)

    ax1.set_title('Predictions vs True Function', fontsize=12)
    ax1.set_xlabel('Test Samples (Sorted by True Class)', fontsize=10)
    ax1.set_ylabel('Regression Output', fontsize=10)
    ax1.set_yticks(np.arange(0, 2.25, 0.25))
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig('q3_stump_vs_bagging_clear.png', dpi=150, bbox_inches='tight')
    

if __name__ == "__main__":
    main()