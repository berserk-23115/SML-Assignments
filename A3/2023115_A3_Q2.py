import numpy as np

def principalComponent(train, var=None, comp=None):
    data_mu = np.mean(train, axis=1, keepdims=True)
    corrected_data_matrix = train - data_mu
    cov_S = np.dot(corrected_data_matrix, corrected_data_matrix.T) / train.shape[1]
    eigVals, vec = np.linalg.eigh(cov_S)
    sorted_idx = np.argsort(eigVals)[::-1]
    eigVals = eigVals[sorted_idx]
    vec = vec[:, sorted_idx]

    if comp is not None:
        k = comp
    elif var is not None:
        total_var = np.sum(eigVals)
        var_sum = 0
        k = 0
        for i in range(len(eigVals)):
            var_sum += eigVals[i]
            if (var_sum / total_var) >= var:
                k = i + 1 
                break
    else:
        k = train.shape[0]
        
    Up = vec[:, :k]
    Y = np.dot(Up.T, corrected_data_matrix)
    return Y, Up, eigVals[:k]


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def weighted_gini_index(y_left, y_right):
    total = len(y_left) + len(y_right)
    if(total == 0):
        return 0.0
    return (len(y_left) / total)*gini_impurity(y_left)+(len(y_right) / total)*gini_impurity(y_right)


def majority_class(y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]


def eval_split(X, y, idx):
    thresh = np.median(X[:, idx])
    feat_vals = X[:, idx]
    left_mask = feat_vals <= thresh
    right_mask = ~left_mask

    if (left_mask.sum() == 0 or right_mask.sum() == 0):
        return None
    y_left = y[left_mask]
    y_right = y[right_mask]
    wgi = weighted_gini_index(y_left, y_right)
    gain = gini_impurity(y) - wgi

    return {
        "feature": idx,
        "threshold": thresh,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "wgi": wgi,
        "gain": gain,
    }


def bestSplitSearch(X, y, idxs):
    best_split = None
    best_wgi = np.inf
    for idx in idxs:
        split_node = eval_split(X, y, idx)
        if split_node is None:
            continue
        if (split_node["wgi"] < best_wgi):
            best_wgi = split_node["wgi"]
            best_split = split_node
    return best_split


class DecisionTree:
    def __init__(self, random_size=None, rng=None):
        self.subsetSize = random_size
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tree_ = None

    def _sample_feats(self, n_feats):
        if (self.subsetSize is None or self.subsetSize >= n_feats):
            return np.arange(n_feats)
        return self.rng.choice(np.arange(n_feats), size=self.subsetSize, replace=False)

    def addleaf(self, y):
        return {"is_leaf": True, "prediction": majority_class(y)}

    def fit(self, X, y):
        n_features = X.shape[1]
        root_features = self._sample_feats(n_features)
        root_split = bestSplitSearch(X, y, root_features)

        if root_split is None:
            self.tree_ = self.addleaf(y)
            return self

        left_idx = np.where(root_split["left_mask"])[0]
        right_idx = np.where(root_split["right_mask"])[0]

        candids = []
        for side_name, idxs in (("left", left_idx), ("right", right_idx)):
            X_child = X[idxs]
            y_child = y[idxs]
            split = None
            gain = -np.inf

            if len(idxs) > 1:
                child_features = self._sample_feats(n_features)
                split = bestSplitSearch(X_child, y_child, child_features)
                if split is not None:
                    gain = split["gain"]

            candids.append({
                "side": side_name,
                "indices": idxs,
                "split": split,
                "gain": gain,
            })

        splt_child = max(candids, key=lambda c: c["gain"])

        root_node = {
            "is_leaf": False,
            "feature": root_split["feature"],
            "threshold": root_split["threshold"],
            "left": self.addleaf(y[left_idx]),
            "right": self.addleaf(y[right_idx]),
        }

        if (splt_child["split"] is not None):
            split_side = splt_child["side"]
            split_indices = splt_child["indices"]
            split_info = splt_child["split"]

            super_lc = split_indices[split_info["left_mask"]]
            super_rc = split_indices[split_info["right_mask"]]

            level_node_2 = {
                "is_leaf": False,
                "feature": split_info["feature"],
                "threshold": split_info["threshold"],
                "left": self.addleaf(y[super_lc]),
                "right": self.addleaf(y[super_rc]),
            }
            root_node[split_side] = level_node_2

        self.tree_ = root_node
        return self

    def _predict_one(self, node, x):
        if node["is_leaf"]:
            return node["prediction"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(node["left"], x)
        return self._predict_one(node["right"], x)

    def predict(self, X):
        return np.array([self._predict_one(self.tree_, row) for row in X])


def bootstrap_idxs(n_samples, boot_cnt, rng):
    bst_sets = []
    for _ in range(boot_cnt):
        s_idxs = rng.integers(0, n_samples, size=n_samples)
        oob_cond = np.ones(n_samples, dtype=bool)
        oob_cond[s_idxs] = False
        oob_indices = np.where(oob_cond)[0]
        bst_sets.append((s_idxs, oob_indices))
    return bst_sets


def majority_vote(predictions_2d):
    voted = []
    for preds_for_point in predictions_2d.T:
        labels, counts = np.unique(preds_for_point, return_counts=True)
        voted.append(labels[np.argmax(counts)])
    return np.array(voted)


def accuracy_report(y_true, y_pred, classes=(0, 1, 2)):
    overall_acc = np.mean(y_true == y_pred)
    class_acc = {}
    for c in classes:
        cond = (y_true == c)
        class_acc[c] = np.mean(y_pred[cond] == y_true[cond])
    return overall_acc, class_acc


def printAccuracy(title, overall_acc, class_acc):
    print(title)
    print(f"Overall accuracy: {overall_acc * 100:.2f}%")
    for c in sorted(class_acc.keys()):
        print(f"Class {c} accuracy: {class_acc[c] * 100:.2f}%")
    print("-" * 30)


def main():
    dataset = np.load('mnist.npz')
    x_train  = dataset['x_train']
    y_train_full = dataset['y_train']
    x_test = dataset['x_test'] 
    y_test_full = dataset['y_test']
    cond_train = (y_train_full == 0) | (y_train_full == 1) | (y_train_full == 2)
    x_012 = x_train[cond_train]
    y_train = y_train_full[cond_train]

    cond_test = (y_test_full == 0) | (y_test_full == 1) | (y_test_full == 2)
    x_test_012 = x_test[cond_test]
    y_test = y_test_full[cond_test]

    x_train_norm = (x_012 / 255.0).reshape(x_012.shape[0], -1)
    x_test_norm = (x_test_012 / 255.0).reshape(x_test_012.shape[0], -1)
    x_train_pca, Up, eigVals = principalComponent(x_train_norm.T, comp=10)
    train_mu = np.mean(x_train_norm.T, axis=1, keepdims=True)
    x_test_pca = np.dot(Up.T, x_test_norm.T - train_mu)
    x_train_pca = x_train_pca.T
    x_test_pca = x_test_pca.T


    print("Data after filtering classes {0,1,2}, flattening, normalization, and PCA (p=10):")
    print(f"Train shape: {x_train_pca.shape}, Test shape: {x_test_pca.shape}")

    intVal = np.random.default_rng(2023115)
    tree = DecisionTree(rng=intVal)
    tree.fit(x_train_pca, y_train)
    task1_preds = tree.predict(x_test_pca)
    task1_overall, task1_class = accuracy_report(y_test, task1_preds)
    printAccuracy("Custom Decision Tree (3 terminal nodes)", task1_overall, task1_class)


    rng = np.random.default_rng(2023115)
    n_trees = 5
    bootstrap_sets = bootstrap_idxs(len(y_train), n_trees, rng)

    bagging_trees = []
    bagging_oob_errors = []
    for i, (boot_idx, oob_idx) in enumerate(bootstrap_sets):
        model = DecisionTree(rng=np.random.default_rng(2023115 + i))
        model.fit(x_train_pca[boot_idx], y_train[boot_idx])
        bagging_trees.append(model)

        if len(oob_idx) > 0:
            oob_pred = model.predict(x_train_pca[oob_idx])
            bagging_oob_errors.append(1.0 - np.mean(oob_pred == y_train[oob_idx]))

    avg_bagging_oob_error = np.mean(bagging_oob_errors)
    print(f"Bagging average OOB error (5 trees): {avg_bagging_oob_error:.4f}")

    bag_preds_test = np.array([model.predict(x_test_pca) for model in bagging_trees])
    bag_final_pred = majority_vote(bag_preds_test)
    bagging_final, bagging_class  = accuracy_report(y_test, bag_final_pred)
    printAccuracy("Bagging Test Performance", bagging_final, bagging_class)

    
    k_values = range(1, 11)
    k_val = {}    
    for k in k_values:
        rf_trees = []
        rf_oob_errors = []
        for i, (boot_idx, oob_idx) in enumerate(bootstrap_sets):
            rf_model = DecisionTree(
                random_size=k, 
                rng=np.random.default_rng(2023115 + i),
            )
            rf_model.fit(x_train_pca[boot_idx], y_train[boot_idx])
            rf_trees.append(rf_model)

            if len(oob_idx) > 0:
                oob_pred = rf_model.predict(x_train_pca[oob_idx])
                error = 1.0 - np.mean(oob_pred == y_train[oob_idx])
                rf_oob_errors.append(error)

        avg_oob = np.mean(rf_oob_errors)
        k_val[k] = avg_oob
        print(f"k={k} , Avg OOB Error: {avg_oob:.4f}")

    best_k = min(k_val, key=k_val.get)
    print(f"\nOptimal k : {best_k} , OOB Error: {k_val[best_k]:.4f})")
    best_rf_trees = []
    for i, (boot_idx, _) in enumerate(bootstrap_sets):
        m = DecisionTree(random_size=best_k, rng=np.random.default_rng(2023115 + i))
        m.fit(x_train_pca[boot_idx], y_train[boot_idx])
        best_rf_trees.append(m)

    rf_test_preds = np.array([model.predict(x_test_pca) for model in best_rf_trees])
    rf_final_pred = majority_vote(rf_test_preds)
    rf_overall, rf_class = accuracy_report(y_test, rf_final_pred)

    printAccuracy(f"Random Forest (Best k={best_k}) Test Performance", rf_overall, rf_class)

if __name__ == "__main__":
    main()