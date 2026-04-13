import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

dataset = np.load('mnist.npz')
x_tr = dataset['x_train']
y_tr = dataset['y_train']
x_te = dataset['x_test']
y_te = dataset['y_test']
print(x_tr.shape)
print(y_tr.shape)
print(x_te.shape)
print(y_te.shape)

def ridgeRegression(X, y, l):
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T @ X + l * I) @ X.T @ y
    return w

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

def mse(yt, yp):
    return np.mean((yt - yp) ** 2)

def predClass(S):
    cls = np.array([0, 1, 2])
    return cls[np.argmax(S, axis=1)]

cm_tr = (y_tr == 0) | (y_tr == 1) | (y_tr == 2)
Xa = x_tr[cm_tr]
ya = y_tr[cm_tr]
Xa = (Xa / 255.0).reshape(Xa.shape[0], -1)

cm_te = (y_te == 0) | (y_te == 1) | (y_te == 2)
Xb = x_te[cm_te]
yb = y_te[cm_te]
Xb = (Xb / 255.0).reshape(Xb.shape[0], -1)

Xp, Up, ev = principalComponent(Xa.T, comp=10)
mu0 = np.mean(Xa.T, axis=1, keepdims=True)
Xq = np.dot(Up.T, Xb.T - mu0)

Xp = Xp.T
Xq = Xq.T

print(Xa.shape)
print(ya.shape)

lv = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
dims = [2, 5, 10, 20, 30]
cls = [0, 1, 2]

Yt = np.column_stack([targ_encod(ya, k) for k in cls])
Ys = np.column_stack([targ_encod(yb, k) for k in cls])

r_tr, r_te = [], []
l_tr, l_te = [], []
l_nz = []
rp1, lp1 = [], []

br_mse = np.inf
bl_mse = np.inf
br_lam = None
bl_lam = None
br_W = None
bl_W = None
bl_b = None

for lam in lv:
    rW = np.zeros((Xp.shape[1], len(cls)))
    for i, k in enumerate(cls):
        rW[:, i] = ridgeRegression(Xp, Yt[:, i], lam)

    rs = Xp @ rW
    rs2 = Xq @ rW
    c1 = mse(Yt, rs)
    c2 = mse(Ys, rs2)
    r_tr.append(c1)
    r_te.append(c2)
    rp1.append(rW[:, 1])

    if (c2 < br_mse):
        br_mse = c2
        br_lam = lam
        br_W = rW.copy()

    lW = np.zeros((Xp.shape[1], len(cls)))
    lb = np.zeros(len(cls))
    for i, k in enumerate(cls):
        m = Lasso(alpha=lam, max_iter=10000)
        m.fit(Xp, Yt[:, i])
        lW[:, i] = m.coef_
        lb[i] = m.intercept_

    ls = Xp @ lW + lb
    ls2 = Xq @ lW + lb
    c3 = mse(Yt, ls)
    c4 = mse(Ys, ls2)
    l_tr.append(c3)
    l_te.append(c4)
    l_nz.append(np.count_nonzero(np.abs(lW) > 1e-8))
    lp1.append(lW[:, 1])

    if (c4 < bl_mse):
        bl_mse = c4
        bl_lam = lam
        bl_W = lW.copy()
        bl_b = lb.copy()

r_tr = np.array(r_tr); r_te = np.array(r_te)
l_tr = np.array(l_tr); l_te = np.array(l_te)
l_nz = np.array(l_nz)
rp1 = np.array(rp1); lp1 = np.array(lp1)

plt.figure(figsize=(9, 6))
plt.plot(lv, r_tr, marker='o', label='Ridge Train MSE')
plt.plot(lv, r_te, marker='o', label='Ridge Test MSE')
plt.plot(lv, l_tr, marker='s', label='Lasso Train MSE')
plt.plot(lv, l_te, marker='s', label='Lasso Test MSE')
plt.xscale('log'); plt.xlabel('lambda (log scale)'); plt.ylabel('MSE')
plt.title('MSE vs lambda (p=10)'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('plot_mse_vs_lambda.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.plot(lv, l_nz, marker='d', color='tab:green')
plt.xscale('log'); plt.xlabel('lambda (log scale)'); plt.ylabel('Non-zero coefficients')
plt.title('Lasso sparsity vs lambda (p=10)'); plt.grid(alpha=0.3)
plt.savefig('plot_lasso_sparsity.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(9, 6))
for j in range(rp1.shape[1]):
    plt.plot(lv, rp1[:, j], linewidth=1)
plt.xscale('log'); plt.xlabel('lambda (log scale)'); plt.ylabel('Coefficient value')
plt.title('Ridge regularization paths (class 1)'); plt.grid(alpha=0.3)
plt.savefig('plot_ridge_regularization_paths.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(9, 6))
for j in range(lp1.shape[1]):
    plt.plot(lv, lp1[:, j], linewidth=1)
plt.xscale('log'); plt.xlabel('lambda (log scale)'); plt.ylabel('Coefficient value')
plt.title('Lasso regularization paths (class 1)'); plt.grid(alpha=0.3)
plt.savefig('plot_lasso_regularization_paths.png', dpi=300, bbox_inches='tight')

r_tr_p, r_te_p = [], []
for p in dims:
    Xpp, Upp, _ = principalComponent(Xa.T, comp=p)
    Xqp = np.dot(Upp.T, Xb.T - mu0)
    Xpp = Xpp.T; Xqp = Xqp.T

    rWp = np.zeros((Xpp.shape[1], len(cls)))
    for i, k in enumerate(cls):
        rWp[:, i] = ridgeRegression(Xpp, Yt[:, i], br_lam)
    r_tr_p.append(mse(Yt, Xpp @ rWp))
    r_te_p.append(mse(Ys, Xqp @ rWp))

plt.figure(figsize=(8, 5))
plt.plot(dims, r_tr_p, marker='o', label='Train MSE')
plt.plot(dims, r_te_p, marker='o', label='Test MSE')
plt.xlabel('PCA dimensions (p)'); plt.ylabel('MSE')
plt.title(f'Ridge error vs model complexity (lambda={br_lam})')
plt.grid(alpha=0.3); plt.legend()
plt.savefig('plot_ridge_model_complexity.png', dpi=300, bbox_inches='tight')

br_pred = predClass(Xq @ br_W)
bl_pred = predClass(Xq @ bl_W + bl_b)
print(f'Best Ridge lambda (p=10): {br_lam}')
print(f'Best Lasso lambda (p=10): {bl_lam}')
print(f'Best Ridge test accuracy: {np.mean(br_pred == yb) * 100:.2f}%')
print(f'Best Lasso test accuracy: {np.mean(bl_pred == yb) * 100:.2f}%')
plt.show()