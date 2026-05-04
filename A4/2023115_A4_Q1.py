import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2023115)

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


def wtError(yl, yr, dl, dr):
    a = np.sum(dl[yl==1])
    b = np.sum(dl[yl==-1])
    a_u = np.sum(dr[yr==1])
    b_u = np.sum(dr[yr==-1])
    if (a >= b):
        lpred = 1
    else:
        lpred = -1
    if (a_u >= b_u):
        rpred = 1
    else:
        rpred = -1
    left_error = np.sum(dl[yl!=lpred])
    right_error = np.sum(dr[yr!=rpred])
    tot_error = left_error+right_error
    return tot_error, lpred, rpred


def train_stump(X, y, D):
    bf, bt = None, None
    best_error = np.inf
    blp, brp = None, None
    for fi in range(X.shape[1]):
        xs = X[:, fi]
        si = np.argsort(xs)
        xs, ys, ds = xs[si], y[si], D[si]
        for i in range(len(xs)-1):
            if (xs[i] == xs[i+1]):
                continue
            thr = (xs[i]+xs[i+1])/2.0
            error, lp, rp = wtError(ys[:i + 1], ys[i+1:],ds[:i + 1], ds[i + 1:])
            if (error<best_error):
                best_error, bf, bt, blp, brp = error, fi, thr, lp, rp
    wErro = best_error / np.sum(D)
    return {'feat': bf,'thresh': bt,'lpred': blp,'rpred': brp,'error': wErro,'mclfs_count': best_error}


def predict_stump(stump, X):
    preds = np.zeros(X.shape[0], dtype=int)
    lm = X[:, stump['feat']] <= stump['thresh']
    preds[lm] = stump['lpred']
    preds[~lm] = stump['rpred']
    return preds


def adaboost_stumps(X_train, y_train, X_val, y_val, X_test, y_test):
    n_stumps=300
    n_samples = X_train.shape[0]
    D = np.ones(n_samples)/n_samples
    stumps = []
    alphas = []
    train_acc = []
    val_acc = []
    test_acc = []
    best_val_acc = -np.inf
    best_iter = 0
    
    print(f"\n{'Iter'} {'Train Acc'} {'Val Acc'} {'Test Acc'} {'α_t'} {'Error'}")
    for t in range(n_stumps):
        stump = train_stump(X_train, y_train, D)
        stumps.append(stump)
        epsilon_t = stump['error']
        alpha_t = 0.5*np.log((1-epsilon_t)/epsilon_t)
        alphas.append(alpha_t)
        h_t = predict_stump(stump, X_train)
        D = D*np.exp(-alpha_t*y_train*h_t)
        D = D/np.sum(D)
        train_pred = predict(stumps, alphas, X_train)
        val_pred = predict(stumps, alphas, X_val)
        test_pred = predict(stumps, alphas, X_test)
        
        acc_train = np.mean(train_pred==y_train)
        acc_val = np.mean(val_pred==y_val)
        acc_test = np.mean(test_pred==y_test)
        
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        test_acc.append(acc_test)
        if (acc_val > best_val_acc):
            best_val_acc = acc_val
            best_iter = t
        if ((t + 1)%50==0):
            print(f"{t+1} {acc_train} {acc_val} {acc_test} {alpha_t} {epsilon_t}")
    
    best_test_acc = test_acc[best_iter]
    print("\n")
    print(f"Best iteration: {best_iter + 1}")
    print(f"Best validation accuracy: {best_val_acc:.6f}")
    print(f"Test accuracy at best iteration: {best_test_acc:.6f}")
    return stumps, np.array(alphas), train_acc, val_acc, test_acc, best_iter, best_test_acc


def predict(stumps, alphas, X):
    n_samples = X.shape[0]
    scores = np.zeros(n_samples)
    for stump, alpha in zip(stumps, alphas):
        h = predict_stump(stump, X)
        scores += alpha * h
    return np.sign(scores).astype(int)


print("MNIST 4 vs 9 Classification with AdaBoost")
dataset = np.load('mnist.npz')
x_train_all = dataset['x_train']
y_train_all = dataset['y_train']
x_test_all = dataset['x_test']
y_test_all = dataset['y_test']

train_mask = (y_train_all==4) | (y_train_all==9)
test_mask = (y_test_all==4) | (y_test_all==9)

x_train_49 = x_train_all[train_mask]
y_train_49 = y_train_all[train_mask]
x_test_49 = x_test_all[test_mask]
y_test_49 = y_test_all[test_mask]

idx_class4 = np.where(y_train_49 == 4)[0]
idx_class9 = np.where(y_train_49 == 9)[0]

val_idx = np.concatenate([idx_class4[:1000], idx_class9[:1000]])
train_idx = np.concatenate([idx_class4[1000:], idx_class9[1000:]])

x_train = x_train_49[train_idx]
y_train = y_train_49[train_idx]
x_val = x_train_49[val_idx]
y_val = y_train_49[val_idx]
x_test = x_test_49
y_test = y_test_49

print(f"\nData shapes before preprocessing:")
print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


x_train_flat = x_train.reshape(x_train.shape[0], -1).T.astype(np.float64)
x_val_flat = x_val.reshape(x_val.shape[0], -1).T.astype(np.float64)
x_test_flat = x_test.reshape(x_test.shape[0], -1).T.astype(np.float64)

print("\nApplying PCA dimension reduction (5 components)...")
Y_train, pca_basis, ev = principalComponent(x_train_flat, comp=5)

mu_train = np.mean(x_train_flat, axis=1, keepdims=True)

Y_val = np.dot(pca_basis.T, x_val_flat - mu_train)
Y_test = np.dot(pca_basis.T, x_test_flat - mu_train)
Y_train = Y_train.T
Y_val = Y_val.T
Y_test = Y_test.T


y_train = targ_encod(y_train, 9)
y_val = targ_encod(y_val, 9)
y_test = targ_encod(y_test, 9)

print(f"\nData shapes after PCA:")
print(f"Train: {Y_train.shape}, Val: {Y_val.shape}, Test: {Y_test.shape}")
print(f"Train labels: {np.unique(y_train)} (class 4 -> -1, class 9 -> +1)")
print(f"Val labels: {np.unique(y_val)}")
print(f"Test labels: {np.unique(y_test)}")

print("Training AdaBoost with Decision Stumps (300 trees)")
stumps, alphas, train_acc, val_acc, test_acc, best_iter, best_test_acc = adaboost_stumps(
    Y_train, y_train, Y_val, y_val, Y_test, y_test, n_stumps=300
)

print(f"Number of stumps grown: {len(stumps)}")
print(f"Best iteration (highest validation accuracy): {best_iter + 1}")
print(f"Best validation accuracy: {val_acc[best_iter]:.6f}")
print(f"Test accuracy at best iteration: {best_test_acc:.6f}")
print(f"\nFinal accuracies (at iteration {best_iter + 1}):")
print(f"Train: {train_acc[best_iter]:.6f}")
print(f"Val:   {val_acc[best_iter]:.6f}")
print(f"Test:  {test_acc[best_iter]:.6f}")

plt.figure(figsize=(12, 7))
plt.plot(range(1, len(val_acc)+1), val_acc, 'b-', label='Validation Accuracy',linewidth=2.5, alpha=0.8)
plt.plot(range(1, len(test_acc)+1), test_acc, 'r-', label='Test Accuracy',linewidth=2.5, alpha=0.8)
plt.axvline(x=best_iter + 1, color='g', linestyle='--', linewidth=2,label=f'Best Iteration ({best_iter + 1})')

plt.xlabel('Number of Trees', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
plt.title('AdaBoost: Validation & Test Accuracy vs Number of Trees',fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0.85, 1.02])
plt.tight_layout()
plt.savefig('adaboost_accuracy.png', dpi=150)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(range(1, len(train_acc) + 1), train_acc, 'g-', label='Train Accuracy',linewidth=2.5, alpha=0.8)
plt.plot(range(1, len(val_acc) + 1), val_acc, 'b-', label='Validation Accuracy',linewidth=2.5, alpha=0.8)
plt.plot(range(1, len(test_acc) + 1), test_acc, 'r-', label='Test Accuracy',linewidth=2.5, alpha=0.8)
plt.axvline(x=best_iter + 1, color='k', linestyle='--', linewidth=2, alpha=0.6,label=f'Best Iteration ({best_iter + 1})')

plt.xlabel('Number of Trees', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
plt.title('AdaBoost: Train/Val/Test Accuracy vs Number of Trees',fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0.85, 1.02])
plt.tight_layout()
plt.savefig('adaboost_accuracy_all_sets.png', dpi=150)
plt.show()
