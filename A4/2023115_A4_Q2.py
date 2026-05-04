import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2023115)
rng = np.random.default_rng(2023115)

def principalComponent(X, var=None, comp=None):
    mu = np.mean(X, axis=1, keepdims=True)
    Xc = X - mu
    S = np.dot(Xc, Xc.T) / X.shape[1]
    ev, V = np.linalg.eigh(S)
    idx = np.argsort(ev)[::-1]
    ev = ev[idx]
    V = V[:, idx]
    if comp is not None:
        k = comp
    elif var is not None:
        tv = np.sum(ev)
        vs = 0
        k = 0
        for i in range(len(ev)):
            vs += ev[i]
            if (vs / tv) >= var:
                k = i + 1
                break
    else:
        k = X.shape[0]
    Up = V[:, :k]
    Y = np.dot(Up.T, Xc)
    return Y, Up, ev[:k]

def targ_encod(y, k):
    return np.where(y == k, 1, -1)

def compute_ssr(yl, yr):
    if (len(yl) == 0 or len(yr) == 0):
        return np.inf, None, None
    lm, rm = np.mean(yl), np.mean(yr)
    ssr = np.sum((yl - lm)**2) + np.sum((yr - rm)**2)
    return ssr, lm, rm


def train_stump(X, y, max_cands=1000):
    rng = np.random.default_rng(2023115)
    bf, bt, bssr, blv, brv = None, None, np.inf, None, None
    for fi in range(X.shape[1]):
        xs = X[:, fi]
        si = np.argsort(xs)
        xs, ys = xs[si], y[si]
        cand_indx = []
        for i in range(len(xs) - 1):
            if xs[i] != xs[i + 1]:
                cand_indx.append(i)

        if len(cand_indx) > max_cands:
            cand_indx = rng.choice(cand_indx, size= max_cands, replace=False)
            cand_indx = sorted(cand_indx)
        for i in cand_indx:
            thr = (xs[i]+xs[i+1]) / 2.0
            ssr, lv, rv = compute_ssr(ys[:i+1], ys[i+1:])
            if (ssr < bssr):
                bssr, bf, bt, blv, brv = ssr, fi, thr, lv, rv
    
    return {'feat': bf, 'thresh': bt, 'lval': blv,'rval': brv, 'ssr': bssr}


def predict_stump(mdl, X):
    preds = np.zeros(X.shape[0])
    lm = X[:, mdl['feat']] <= mdl['thresh']
    preds[lm] = mdl['lval']
    preds[~lm] = mdl['rval']
    return preds


def samplesBootStrap(X, y, n, rng):
    ns = X.shape[0]
    out = []
    for _ in range(n):
        si = rng.integers(0, ns, size=ns)
        oob = np.where(~np.isin(np.arange(ns), si))[0]
        out.append((si, oob))
    return out

def gradBoosting(X_train, y_train, X_val, y_val, X_test, y_test,eta=0.01):
    stumps = []
    n_stumps = 300
    F_train = np.zeros(X_train.shape[0])
    F_val = np.zeros(X_val.shape[0])
    F_test = np.zeros(X_test.shape[0])
    train_mse = []
    val_mse = []
    test_mse = []
    best_val_mse = float('inf')
    best_iter = 0
    
    for i in range(n_stumps):
        residuals_train = np.sign(y_train - F_train)
        stump = train_stump(X_train, residuals_train)
        stumps.append(stump)
        h_train = predict_stump(stump, X_train)
        h_val = predict_stump(stump, X_val)
        h_test = predict_stump(stump, X_test)
        F_train += eta * h_train
        F_val += eta * h_val
        F_test += eta * h_test
        
        mse_train = np.mean((y_train - F_train) ** 2)
        mse_val = np.mean((y_val - F_val) ** 2)
        mse_test = np.mean((y_test - F_test) ** 2)
        
        train_mse.append(mse_train)
        val_mse.append(mse_val)
        test_mse.append(mse_test)

        if (mse_val < best_val_mse):
            best_val_mse = mse_val
            best_iter = i

        if ((i+1)%50 == 0):
            print(f"Iteration {i+1:3d}: Train MSE={mse_train:.6f}, Val MSE={mse_val:.6f}, Test MSE={mse_test:.6f}")
    
    best_test_mse = test_mse[best_iter]
    return stumps, train_mse, val_mse, test_mse, best_iter, best_test_mse


def predict_ensemble(stumps, X, eta):
    F = np.zeros(X.shape[0])
    for stump in stumps:
        F += eta*predict_stump(stump, X)
    return F



dataset = np.load('mnist.npz')
x_tr_gb = dataset['x_train']
y_tr_lab = dataset['y_train']
x_test_all = dataset['x_test']
y_test_all = dataset['y_test']

train_mask = (y_tr_lab==4)|(y_tr_lab== 9)
test_mask = (y_test_all==4)|(y_test_all==9)

x_tr_sort = x_tr_gb[train_mask]
y_tr_sort = y_tr_lab[train_mask]
x_te_sort = x_test_all[test_mask]
y_te_sort = y_test_all[test_mask]

idx_cs4 = np.where(y_tr_sort == 4)[0]
idx_cs9 = np.where(y_tr_sort== 9)[0]

val_idx = np.concatenate([idx_cs4[:1000], idx_cs9[:1000]])
train_idx = np.concatenate([idx_cs4[1000:], idx_cs9[1000:]])

x_train = x_tr_sort[train_idx]
y_train = y_tr_sort[train_idx]
x_val = x_tr_sort[val_idx]
y_val = y_tr_sort[val_idx]
x_test = x_te_sort
y_test = y_te_sort


x_tr_flat = x_train.reshape(x_train.shape[0], -1).T.astype(np.float64)
x_val_flat = x_val.reshape(x_val.shape[0], -1).T.astype(np.float64)
x_test_flat = x_test.reshape(x_test.shape[0], -1).T.astype(np.float64)


Y_train, pca_basis, ev = principalComponent(x_tr_flat, comp=5)
mu_train = np.mean(x_tr_flat, axis=1, keepdims=True)

Y_val = np.dot(pca_basis.T, x_val_flat - mu_train)
Y_test = np.dot(pca_basis.T, x_test_flat - mu_train)

Y_train = Y_train.T
Y_val = Y_val.T
Y_test = Y_test.T

y_train = targ_encod(y_train, 9)
y_val = targ_encod(y_val, 9)
y_test = targ_encod(y_test, 9)

print(f"Train shape: {Y_train.shape}, Val shape: {Y_val.shape}, Test shape: {Y_test.shape}")
print(f"Train labels: {np.unique(y_train)}, Val labels: {np.unique(y_val)}, Test labels: {np.unique(y_test)}")

print("Testing with LR = 0.01")
stumps, train_mse, val_mse, test_mse, best_iter, best_test_mse = gradBoosting(
    Y_train, y_train, Y_val, y_val, Y_test, y_test,
    eta=0.01)

print(f"\nBest iteration: {best_iter + 1}")
print(f"Best validation MSE: {val_mse[best_iter]:.6f}")
print(f"Test MSE at best iteration: {best_test_mse:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(val_mse) + 1), val_mse, 'b-', label='Validation MSE', linewidth=2)
plt.axvline(x=best_iter + 1, color='r', linestyle='--', label=f'Best iteration ({best_iter + 1})')
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Validation MSE vs Number of Trees (η = 0.01)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_mse_eta001.png', dpi=150)
print("\nPlot saved as 'validation_mse_eta001.png'")
plt.show()


learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
results = {}
print("Testing multiple learning rates")

for lr in learning_rates:
    print(f"\nTesting η = {lr}")
    obj = {}
    stumps_eta, train_mse_eta, val_mse_eta, test_mse_eta, best_iter_eta, best_test_mse_eta = gradBoosting(Y_train, y_train, Y_val, y_val, Y_test, y_test,eta=lr)
    
    results[lr] = {
        'stumps': stumps_eta,
        'train_mse': train_mse_eta,
        'val_mse': val_mse_eta,
        'test_mse': test_mse_eta,
        'best_iter': best_iter_eta,
        'best_test_mse': best_test_mse_eta,
        'best_val_mse': val_mse_eta[best_iter_eta]
    }
    
    print(f"Best iteration: {best_iter_eta + 1}")
    print(f"Best val MSE: {results[lr]['best_val_mse']:.6f}")
    print(f"Test MSE at best: {best_test_mse_eta:.6f}")


print("SUMMARY TABLE")

print(f"{'Learning Rate':<15} {'Best Iter':<12} {'Best Val MSE':<15} {'Test MSE':<15}")

for eta in learning_rates:
    r = results[eta]
    best_iter_num = r["best_iter"] + 1
    best_val_mse = r["best_val_mse"]
    best_test_mse = r["best_test_mse"]
    print(
        f"{eta:<15} "
        f"{best_iter_num} "
        f"{best_val_mse} "
        f"{best_test_mse}"
    )

plt.figure(figsize=(12, 7))
for eta in learning_rates:
    plt.plot(range(1, len(results[eta]['val_mse']) + 1), results[eta]['val_mse'],label=f'η = {eta}', linewidth=2, alpha=0.8)

plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Validation MSE', fontsize=12)
plt.title('Validation MSE vs Number of Trees (Multiple Learning Rates)', fontsize=14)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_mse_all_etas.png', dpi=150)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, eta in enumerate(learning_rates):
    ax = axes[idx]
    r = results[eta]
    ax.plot(range(1,len(r['train_mse'])+1), r['train_mse'], 'g-', label='Train MSE', alpha=0.7)
    ax.plot(range(1,len(r['val_mse'])+1), r['val_mse'], 'b-', label='Validation MSE', alpha=0.7)
    ax.plot(range(1,len(r['test_mse'])+1), r['test_mse'], 'r-', label='Test MSE', alpha=0.7)
    ax.axvline(x=r['best_iter']+1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Trees', fontsize=10)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_title(f'η = {eta}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mse_comparison_all_etas.png', dpi=150)
plt.show()