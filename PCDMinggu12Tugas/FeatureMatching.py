"""
Sistem Pencocokan Objek Berbasis Fitur Lokal
Implementasi: SIFT, ORB, Feature Matching, BoVW, PCA
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_predict
import itertools

np.random.seed(42)

# ============================================================
# BAGIAN 1: PEMBUATAN DATASET SINTETIS
# ============================================================

OBJECTS = ['buku', 'mug', 'botol', 'mainan', 'remote']
COLORS  = {
    'buku':   [(60, 30, 10),   (80, 50, 20)],
    'mug':    [(180, 80, 60),  (200, 100, 80)],
    'botol':  [(40, 100, 60),  (60, 130, 80)],
    'mainan': [(200, 160, 20), (220, 180, 40)],
    'remote': [(30, 30, 30),   (60, 60, 60)],
}

def draw_object(obj_name, size=200):
    """Gambar objek sintetis unik untuk tiap kategori"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    c1, c2 = COLORS[obj_name]
    cx, cy = size//2, size//2

    if obj_name == 'buku':
        # Buku: persegi panjang dengan garis halaman
        cv2.rectangle(img, (40, 30), (160, 170), c1[::-1], -1)
        cv2.rectangle(img, (40, 30), (160, 170), (0,0,0), 2)
        cv2.rectangle(img, (48, 38), (72, 162), c2[::-1], -1)
        for y in range(50, 160, 12):
            cv2.line(img, (80, y), (152, y), (180,180,180), 1)
        cv2.putText(img, 'BOOK', (82, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    elif obj_name == 'mug':
        # Mug: ellipse + handle
        cv2.ellipse(img, (cx, cy+10), (55, 70), 0, 0, 360, c1[::-1], -1)
        cv2.ellipse(img, (cx, cy+10), (55, 70), 0, 0, 360, (0,0,0), 2)
        cv2.ellipse(img, (cx, 30), (50, 20), 0, 0, 360, c2[::-1], -1)
        # Handle
        pts = np.array([[155,70],[175,70],[175,120],[155,120]], np.int32)
        cv2.polylines(img, [pts], True, (0,0,0), 3)

    elif obj_name == 'botol':
        # Botol: silinder dengan leher
        cv2.rectangle(img, (70, 70), (130, 170), c1[::-1], -1)
        cv2.rectangle(img, (70, 70), (130, 170), (0,0,0), 2)
        cv2.rectangle(img, (85, 30), (115, 70), c2[::-1], -1)
        cv2.rectangle(img, (85, 30), (115, 70), (0,0,0), 2)
        cv2.rectangle(img, (80, 20), (120, 35), (200,200,200), -1)
        cv2.rectangle(img, (80, 20), (120, 35), (0,0,0), 2)
        cv2.line(img, (80, 110), (130, 110), (255,255,255), 2)

    elif obj_name == 'mainan':
        # Mobil mainan: badan + roda
        cv2.rectangle(img, (30, 100), (170, 155), c1[::-1], -1)
        cv2.rectangle(img, (30, 100), (170, 155), (0,0,0), 2)
        pts = np.array([[60,100],[80,60],[140,60],[160,100]], np.int32)
        cv2.fillPoly(img, [pts], c2[::-1])
        cv2.polylines(img, [pts], True, (0,0,0), 2)
        for wx in [60, 140]:
            cv2.circle(img, (wx, 155), 20, (50,50,50), -1)
            cv2.circle(img, (wx, 155), 12, (100,100,100), -1)
        # Jendela
        cv2.rectangle(img, (85, 65), (135, 98), (150,220,255), -1)

    elif obj_name == 'remote':
        # Remote: persegi panjang memanjang dengan tombol
        cv2.rectangle(img, (65, 15), (135, 185), c1[::-1], -1)
        cv2.rectangle(img, (65, 15), (135, 185), (0,0,0), 2)
        # Power button
        cv2.circle(img, (100, 45), 14, (200,50,50), -1)
        # Tombol-tombol
        for r in range(3):
            for c_ in range(3):
                bx = 78 + c_*20
                by = 80 + r*26
                cv2.rectangle(img, (bx, by), (bx+14, by+18), c2[::-1], -1)
                cv2.rectangle(img, (bx, by), (bx+14, by+18), (0,0,0), 1)

    return img


def apply_transform(img, transform_type, param=1.0):
    """Terapkan berbagai transformasi untuk membuat variasi citra uji"""
    h, w = img.shape[:2]

    if transform_type == 'rotation':
        angle = param  # derajat
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        result = cv2.warpAffine(img, M, (w, h), borderValue=(240,240,240))

    elif transform_type == 'scale':
        scale = param
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w, new_h))
        result = np.ones((h, w, 3), dtype=np.uint8) * 240
        # Tempel di tengah
        y_off = max(0, (h - new_h)//2)
        x_off = max(0, (w - new_w)//2)
        cy = min(new_h, h - y_off)
        cx = min(new_w, w - x_off)
        result[y_off:y_off+cy, x_off:x_off+cx] = resized[:cy, :cx]

    elif transform_type == 'illumination':
        factor = param  # < 1 gelap, > 1 terang
        result = np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)

    elif transform_type == 'occlusion':
        result = img.copy()
        # Tutup sebagian dengan kotak abu-abu
        occ_size = int(param)
        rx = np.random.randint(0, w - occ_size)
        ry = np.random.randint(0, h - occ_size)
        result[ry:ry+occ_size, rx:rx+occ_size] = 200

    elif transform_type == 'noise':
        noise = np.random.normal(0, param, img.shape).astype(np.int16)
        result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    else:
        result = img.copy()

    return result


def create_dataset():
    """Buat dataset: 5 objek × (1 ref + 4 uji) = 25 citra"""
    dataset = {}
    transforms = [
        ('rotation',    30),
        ('scale',       0.7),
        ('illumination',0.5),
        ('occlusion',   80),
    ]

    for obj in OBJECTS:
        ref = draw_object(obj, 200)
        tests = []
        for t_type, t_param in transforms:
            t_img = apply_transform(ref, t_type, t_param)
            tests.append((t_img, t_type))
        dataset[obj] = {'ref': ref, 'tests': tests}

    return dataset


def visualize_dataset(dataset, save_path='fig_dataset.png'):
    fig, axes = plt.subplots(5, 5, figsize=(16, 17))
    fig.suptitle('Dataset: 5 Objek × (1 Referensi + 4 Varian Transformasi)', fontsize=13, fontweight='bold')

    transform_labels = ['Referensi', 'Rotasi 30°', 'Skala 0.7×', 'Iluminasi 0.5×', 'Oklusi']

    for row, obj in enumerate(OBJECTS):
        ref = dataset[obj]['ref']
        axes[row][0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        axes[row][0].set_title(f'{obj.upper()}\n{transform_labels[0]}', fontsize=8, fontweight='bold')
        axes[row][0].axis('off')

        for col, (t_img, t_type) in enumerate(dataset[obj]['tests'], start=1):
            axes[row][col].imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(transform_labels[col], fontsize=8)
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# BAGIAN 2: DETEKSI & DESKRIPSI FITUR
# ============================================================

def extract_features(img_bgr, method='SIFT'):
    """Ekstrak keypoints dan descriptors"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    if method == 'SIFT':
        det = cv2.SIFT_create(nfeatures=500)
        kps, desc = det.detectAndCompute(gray, None)
    elif method == 'ORB':
        det = cv2.ORB_create(nfeatures=500)
        kps, desc = det.detectAndCompute(gray, None)
    elapsed = (time.perf_counter() - t0) * 1000

    desc_dim = desc.shape[1] if desc is not None and len(desc) > 0 else 0
    n_kps = len(kps)
    return kps, desc, elapsed, n_kps, desc_dim


def visualize_keypoints(dataset, save_path='fig_keypoints.png'):
    """Visualisasi keypoints SIFT dan ORB pada citra referensi"""
    fig, axes = plt.subplots(5, 2, figsize=(12, 22))
    fig.suptitle('Deteksi Keypoints: SIFT vs ORB\n(pada citra referensi tiap objek)', fontsize=13, fontweight='bold')

    all_stats = []
    for row, obj in enumerate(OBJECTS):
        ref = dataset[obj]['ref']
        for col, method in enumerate(['SIFT', 'ORB']):
            kps, desc, elapsed, n_kps, desc_dim = extract_features(ref, method)
            img_kp = cv2.drawKeypoints(ref, kps, None,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                        color=(0, 200, 0) if method == 'SIFT' else (0, 100, 255))
            axes[row][col].imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(
                f'{obj.upper()} – {method}\n{n_kps} kpts | {desc_dim}D | {elapsed:.1f}ms',
                fontsize=8, fontweight='bold')
            axes[row][col].axis('off')
            all_stats.append({'obj': obj, 'method': method, 'n_kps': n_kps,
                               'desc_dim': desc_dim, 'time_ms': round(elapsed, 2)})

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")
    return all_stats


# ============================================================
# BAGIAN 3: FEATURE MATCHING
# ============================================================

def match_features(desc1, desc2, method='SIFT', matcher_type='BF'):
    """Cocokkan fitur dengan BF atau FLANN, terapkan Lowe's ratio test"""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return [], [], 0

    t0 = time.perf_counter()

    if matcher_type == 'BF':
        if method == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)

    else:  # FLANN
        if method == 'ORB':
            # FLANN dengan LSH untuk binary descriptors
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            desc1 = desc1.astype(np.uint8)
            desc2 = desc2.astype(np.uint8)
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1.astype(np.float32) if method != 'ORB' else desc1,
                                     desc2.astype(np.float32) if method != 'ORB' else desc2, k=2)
        except:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING if method == 'ORB' else cv2.NORM_L2)
            matches = matcher.knnMatch(desc1, desc2, k=2)

    elapsed = (time.perf_counter() - t0) * 1000

    # Lowe's ratio test
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    return good, matches, elapsed


def ransac_homography(kps1, kps2, good_matches):
    """RANSAC untuk estimasi homography dan filter inlier"""
    if len(good_matches) < 4:
        return None, good_matches, []

    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return H, good_matches, []
    inliers  = [m for m, msk in zip(good_matches, mask.ravel()) if msk]
    outliers = [m for m, msk in zip(good_matches, mask.ravel()) if not msk]
    return H, inliers, outliers


def visualize_matching(dataset, save_path='fig_matching.png'):
    """Visualisasi hasil feature matching (inlier vs outlier)"""
    fig, axes = plt.subplots(5, 2, figsize=(18, 25))
    fig.suptitle('Feature Matching: SIFT+BF (kiri) vs ORB+FLANN (riga)\n'
                 'Hijau=inlier, Merah=outlier setelah RANSAC', fontsize=12, fontweight='bold')

    results = []
    for row, obj in enumerate(OBJECTS):
        ref  = dataset[obj]['ref']
        test = dataset[obj]['tests'][0][0]  # rotasi 30°

        for col, (method, mtype) in enumerate([('SIFT','BF'), ('ORB','BF')]):
            kps1, desc1, _, _, _ = extract_features(ref, method)
            kps2, desc2, _, _, _ = extract_features(test, method)

            good, _, t_match = match_features(desc1, desc2, method, mtype)
            H, inliers, outliers = ransac_homography(kps1, kps2, good)

            # Gambar matches
            draw_params_in = dict(matchColor=(0,220,0), singlePointColor=None, flags=2)
            draw_params_out = dict(matchColor=(0,0,220), singlePointColor=None, flags=2)

            img_match = cv2.drawMatches(ref, kps1, test, kps2, inliers[:20], None, **draw_params_in)
            if outliers:
                img_match = cv2.drawMatches(ref, kps1, test, kps2, outliers[:8], img_match,
                                             matchColor=(0,0,220), singlePointColor=None, flags=2+8)

            axes[row][col].imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(
                f'{obj.upper()} | {method}+{mtype}\n'
                f'Good: {len(good)} | Inlier: {len(inliers)} | Outlier: {len(outliers)}',
                fontsize=8, fontweight='bold')
            axes[row][col].axis('off')

            prec = len(inliers)/(len(good)+1e-9)*100
            results.append({'obj': obj, 'method': f'{method}+{mtype}',
                            'good': len(good), 'inliers': len(inliers),
                            'precision': round(prec, 1), 'time_ms': round(t_match,2)})

    plt.tight_layout()
    plt.savefig(save_path, dpi=95, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")
    return results


# ============================================================
# BAGIAN 4: BAG OF VISUAL WORDS
# ============================================================

def build_bovw(dataset, vocab_sizes=[10, 20, 50, 100]):
    """Bangun BoVW vocabulary dengan berbagai ukuran k"""
    print("  Mengekstrak semua fitur SIFT untuk BoVW...")
    all_descs = []
    labels = []
    images_descs = []

    for label_idx, obj in enumerate(OBJECTS):
        ref = dataset[obj]['ref']
        _, desc, _, _, _ = extract_features(ref, 'SIFT')
        if desc is not None and len(desc) > 0:
            all_descs.append(desc)
            labels.append(label_idx)
            images_descs.append(desc)

        for t_img, _ in dataset[obj]['tests']:
            _, desc, _, _, _ = extract_features(t_img, 'SIFT')
            if desc is not None and len(desc) > 0:
                all_descs.append(desc)
                labels.append(label_idx)
                images_descs.append(desc)

    all_descs_flat = np.vstack(all_descs).astype(np.float32)
    print(f"    Total descriptors: {len(all_descs_flat)}")

    results = {}
    for k in vocab_sizes:
        print(f"    K-means k={k}...")
        t0 = time.perf_counter()
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
        kmeans.fit(all_descs_flat)
        t_cluster = (time.perf_counter() - t0) * 1000

        # Buat histogram BoVW untuk tiap citra
        histograms = []
        for desc in images_descs:
            if desc is None or len(desc) == 0:
                histograms.append(np.zeros(k))
                continue
            assignments = kmeans.predict(desc.astype(np.float32))
            hist, _ = np.histogram(assignments, bins=np.arange(k+1))
            hist = hist.astype(float) / (hist.sum() + 1e-9)
            histograms.append(hist)

        X = np.array(histograms)
        y = np.array(labels)

        # Klasifikasi dengan SVM
        if len(X) > 5:
            clf_svm = SVC(kernel='rbf', C=10, random_state=42)
            try:
                y_pred_svm = cross_val_predict(clf_svm, X, y, cv=min(3, len(set(y))))
                acc_svm = (y_pred_svm == y).mean() * 100
            except:
                acc_svm = 0

            # k-NN
            clf_knn = KNeighborsClassifier(n_neighbors=3)
            try:
                y_pred_knn = cross_val_predict(clf_knn, X, cv=min(3, len(set(y))))
                acc_knn = (y_pred_knn == y).mean() * 100
            except:
                acc_knn = 0
        else:
            acc_svm, acc_knn = 0, 0

        results[k] = {
            'kmeans': kmeans, 'histograms': X, 'labels': y,
            'acc_svm': acc_svm, 'acc_knn': acc_knn,
            'time_ms': t_cluster, 'y_pred_svm': y_pred_svm if len(X) > 5 else y
        }

    return results


def visualize_bovw(bovw_results, dataset, save_path='fig_bovw.png'):
    """Visualisasi BoVW: akurasi vs vocab size + confusion matrix"""
    vocab_sizes = sorted(bovw_results.keys())
    acc_svm = [bovw_results[k]['acc_svm'] for k in vocab_sizes]
    acc_knn = [bovw_results[k]['acc_knn'] for k in vocab_sizes]
    times   = [bovw_results[k]['time_ms'] for k in vocab_sizes]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Bag of Visual Words (BoVW) – Analisis Vocabulary Size', fontsize=13, fontweight='bold')

    # Akurasi vs k
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(vocab_sizes, acc_svm, 'o-', color='#2E75B6', lw=2, ms=8, label='SVM (RBF)')
    ax1.plot(vocab_sizes, acc_knn, 's--', color='#E74C3C', lw=2, ms=8, label='k-NN (k=3)')
    ax1.set_xlabel('Vocabulary Size (k)')
    ax1.set_ylabel('Akurasi (%)')
    ax1.set_title('Akurasi Klasifikasi vs Vocab Size')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(vocab_sizes)
    ax1.set_ylim(0, 105)

    # Waktu clustering
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(vocab_sizes, times, color=['#1F3864','#2E75B6','#5B9BD5','#9DC3E6'],
                   edgecolor='white', width=6)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                 f'{t:.0f}ms', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Vocabulary Size (k)')
    ax2.set_ylabel('Waktu (ms)')
    ax2.set_title('Waktu K-Means Clustering')
    ax2.set_xticks(vocab_sizes); ax2.grid(axis='y', alpha=0.3)

    # Histogram visual words (k=50)
    ax3 = fig.add_subplot(gs[0, 2])
    k50 = bovw_results[50]
    colors_obj = ['#E74C3C','#2E75B6','#27AE60','#F39C12','#9B59B6']
    for i, obj in enumerate(OBJECTS):
        obj_hists = k50['histograms'][k50['labels'] == i]
        if len(obj_hists):
            ax3.plot(obj_hists[0], alpha=0.7, lw=1.5, label=obj.capitalize(), color=colors_obj[i])
    ax3.set_xlabel('Visual Word Index')
    ax3.set_ylabel('Frekuensi Ternormalisasi')
    ax3.set_title('Histogram Visual Words (k=50)')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # Confusion matrix terbaik (k=100, SVM)
    best_k = max(vocab_sizes)
    ax4 = fig.add_subplot(gs[1, :2])
    cm = confusion_matrix(bovw_results[best_k]['labels'], bovw_results[best_k]['y_pred_svm'])
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax4)
    ax4.set_xticks(range(5)); ax4.set_yticks(range(5))
    ax4.set_xticklabels([o.capitalize() for o in OBJECTS], rotation=30, ha='right')
    ax4.set_yticklabels([o.capitalize() for o in OBJECTS])
    ax4.set_xlabel('Prediksi'); ax4.set_ylabel('Aktual')
    ax4.set_title(f'Confusion Matrix – BoVW SVM (k={best_k}, acc={bovw_results[best_k]["acc_svm"]:.1f}%)')
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i,j] > thresh else 'black', fontsize=11, fontweight='bold')

    # Precision-Recall proxy (per kelas)
    ax5 = fig.add_subplot(gs[1, 2])
    for i, obj in enumerate(OBJECTS):
        y_true = (bovw_results[50]['labels'] == i).astype(int)
        y_score = bovw_results[50]['histograms'][:, i % bovw_results[50]['histograms'].shape[1]]
        if y_true.sum() > 0 and len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax5.plot(rec, prec, lw=1.5, label=f'{obj.capitalize()} (AP={ap:.2f})', color=colors_obj[i])
    ax5.set_xlabel('Recall'); ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Curve (BoVW, k=50)')
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3)
    ax5.set_xlim([0,1]); ax5.set_ylim([0,1.05])

    plt.savefig(save_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# BAGIAN 5: PCA REDUKSI DIMENSI
# ============================================================

def pca_analysis(dataset, save_path='fig_pca.png'):
    """PCA pada descriptors SIFT/ORB, evaluasi akurasi vs komponen"""
    # Kumpulkan semua descriptors dengan label
    all_sift, all_orb, all_labels = [], [], []
    for label_idx, obj in enumerate(OBJECTS):
        imgs = [dataset[obj]['ref']] + [t for t,_ in dataset[obj]['tests']]
        for img in imgs:
            _, d_sift, _, _, _ = extract_features(img, 'SIFT')
            _, d_orb,  _, _, _ = extract_features(img, 'ORB')
            if d_sift is not None and len(d_sift) > 0:
                all_sift.append(d_sift.mean(axis=0))
                all_labels.append(label_idx)
            if d_orb is not None and len(d_orb) > 0:
                all_orb.append(d_orb.astype(float).mean(axis=0))

    X_sift = np.array(all_sift)
    X_orb  = np.array(all_orb)
    y = np.array(all_labels)

    n_components_list = [8, 16, 32, 64, 128]
    pca_results = {'SIFT': {}, 'ORB': {}}

    for method, X in [('SIFT', X_sift), ('ORB', X_orb)]:
        max_comp = min(X.shape[1], X.shape[0] - 1, 128)
        for n in n_components_list:
            if n > max_comp:
                continue
            pca = PCA(n_components=n)
            X_pca = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_.sum() * 100

            # Matching accuracy: cosine similarity
            correct = 0
            for i in range(len(X_pca)):
                dists = np.linalg.norm(X_pca - X_pca[i], axis=1)
                dists[i] = np.inf
                nearest = np.argmin(dists)
                if y[nearest] == y[i]:
                    correct += 1
            acc = correct / len(X_pca) * 100

            pca_results[method][n] = {'explained': explained, 'acc': acc}

    # Visualisasi
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PCA untuk Reduksi Dimensi Descriptor', fontsize=13, fontweight='bold')

    colors_method = {'SIFT': '#2E75B6', 'ORB': '#E74C3C'}

    # Akurasi vs komponen
    ax = axes[0][0]
    for method in ['SIFT', 'ORB']:
        comps = sorted(pca_results[method].keys())
        accs  = [pca_results[method][c]['acc'] for c in comps]
        ax.plot(comps, accs, 'o-', color=colors_method[method], lw=2, ms=8, label=method)
    ax.set_xlabel('PCA Components')
    ax.set_ylabel('Matching Accuracy (%)')
    ax.set_title('Akurasi vs Jumlah Komponen PCA')
    ax.legend(); ax.grid(alpha=0.3)

    # Variance explained
    ax = axes[0][1]
    for method in ['SIFT', 'ORB']:
        comps = sorted(pca_results[method].keys())
        expl  = [pca_results[method][c]['explained'] for c in comps]
        ax.plot(comps, expl, 's--', color=colors_method[method], lw=2, ms=8, label=method)
    ax.set_xlabel('PCA Components')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Variance Explained vs Komponen PCA')
    ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(95, color='gray', ls=':', label='95% threshold')

    # Compression ratio
    ax = axes[0][2]
    orig_dims = {'SIFT': 128, 'ORB': 32}
    for method in ['SIFT', 'ORB']:
        comps = sorted(pca_results[method].keys())
        ratios = [orig_dims[method] / c for c in comps]
        accs   = [pca_results[method][c]['acc'] for c in comps]
        sc = ax.scatter(ratios, accs, c=[pca_results[method][c]['explained'] for c in comps],
                        cmap='coolwarm', s=120, label=method,
                        marker='o' if method=='SIFT' else 's', zorder=5)
        for ratio, acc, c in zip(ratios, accs, comps):
            ax.annotate(f'{c}D', (ratio, acc), textcoords='offset points',
                        xytext=(5,5), fontsize=7, color=colors_method[method])
    plt.colorbar(sc, ax=ax, label='Variance (%)')
    ax.set_xlabel('Compression Ratio (original/pca)')
    ax.set_ylabel('Matching Accuracy (%)')
    ax.set_title('Akurasi vs Rasio Kompresi')
    ax.legend(); ax.grid(alpha=0.3)

    # 2D PCA scatter (SIFT, 2 komponen)
    ax = axes[1][0]
    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X_sift)
    scatter_colors = ['#E74C3C','#2E75B6','#27AE60','#F39C12','#9B59B6']
    for i, obj in enumerate(OBJECTS):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=scatter_colors[i],
                   label=obj.capitalize(), s=60, alpha=0.8, edgecolors='white', lw=0.5)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA 2D – Distribusi SIFT Descriptors')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Scree plot
    ax = axes[1][1]
    pca_full = PCA()
    pca_full.fit(X_sift)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar)+1), cumvar, '-', color='#2E75B6', lw=2)
    ax.axhline(95, color='red', ls='--', label='95%')
    ax.axhline(99, color='orange', ls='--', label='99%')
    ax.set_xlabel('Komponen PCA')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('Scree Plot – SIFT Descriptors')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, min(80, len(cumvar)))

    # Tabel perbandingan komponen
    ax = axes[1][2]
    ax.axis('off')
    rows = []
    for method in ['SIFT', 'ORB']:
        orig = orig_dims[method]
        for n in sorted(pca_results[method].keys()):
            r = pca_results[method][n]
            rows.append([method, f'{n}D', f'{orig}D', f'{r["acc"]:.1f}%', f'{r["explained"]:.1f}%'])
    col_labels = ['Metode', 'PCA Dim', 'Original', 'Akurasi', 'Var%']
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1F3864')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EBF3FB')
    ax.set_title('Tabel PCA: Akurasi vs Kompresi', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")
    return pca_results


# ============================================================
# BAGIAN 6: EVALUASI KOMPREHENSIF
# ============================================================

def comprehensive_evaluation(dataset, save_path='fig_evaluation.png'):
    """Tabel perbandingan kecepatan, akurasi, robustness"""
    transforms = ['rotation', 'scale', 'illumination', 'occlusion']
    methods = [('SIFT', 'BF'), ('SIFT', 'FLANN'), ('ORB', 'BF')]

    eval_matrix = {}
    for obj in OBJECTS:
        ref = dataset[obj]['ref']
        for method, mtype in methods:
            kps1, desc1, t_ext, nkps, ddim = extract_features(ref, method)
            key = f'{method}+{mtype}'
            if key not in eval_matrix:
                eval_matrix[key] = {t: [] for t in transforms}
                eval_matrix[key]['extract_time'] = []
                eval_matrix[key]['n_kps'] = []

            eval_matrix[key]['extract_time'].append(t_ext)
            eval_matrix[key]['n_kps'].append(nkps)

            for t_img, t_type in dataset[obj]['tests']:
                kps2, desc2, _, _, _ = extract_features(t_img, method)
                good, _, t_match = match_features(desc1, desc2, method, mtype)
                _, inliers, _ = ransac_homography(kps1, kps2, good)
                prec = len(inliers) / (len(good) + 1e-9) * 100
                eval_matrix[key][t_type].append(prec)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Evaluasi Komprehensif – Perbandingan Metode', fontsize=13, fontweight='bold')

    method_keys = [f'{m}+{mt}' for m, mt in methods]
    method_colors = ['#2E75B6', '#1F3864', '#E74C3C']

    # Robustness per transform
    ax = axes[0][0]
    x = np.arange(len(transforms))
    w = 0.25
    for i, (key, color) in enumerate(zip(method_keys, method_colors)):
        means = [np.mean(eval_matrix[key][t]) for t in transforms]
        bars = ax.bar(x + i*w, means, w, label=key, color=color, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([t.capitalize() for t in transforms])
    ax.set_ylabel('Inlier Precision (%)')
    ax.set_title('Robustness per Transformasi')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Waktu ekstraksi
    ax = axes[0][1]
    avg_times = [np.mean(eval_matrix[k]['extract_time']) for k in method_keys]
    bars = ax.barh(method_keys, avg_times, color=method_colors, alpha=0.85)
    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                f'{t:.1f}ms', va='center', fontsize=9)
    ax.set_xlabel('Waktu Rata-rata (ms)')
    ax.set_title('Kecepatan Ekstraksi Fitur')
    ax.grid(axis='x', alpha=0.3)

    # Radar / spider chart – robustness di 4 aspek
    ax = axes[1][0]
    categories = ['Rotasi', 'Skala', 'Iluminasi', 'Oklusi']
    N = len(categories)
    angles = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(2, 2, 3, polar=True)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 100)
    ax.set_title('Radar: Robustness per Aspek', pad=20)

    for key, color in zip(method_keys, method_colors):
        vals = [np.mean(eval_matrix[key][t]) for t in transforms]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=color, lw=2, ms=5, label=key)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)

    # Tabel ringkasan
    ax2 = axes[1][1]
    ax2.axis('off')
    table_data = []
    for key in method_keys:
        avg_rob = np.mean([np.mean(eval_matrix[key][t]) for t in transforms])
        avg_time = np.mean(eval_matrix[key]['extract_time'])
        avg_kps  = np.mean(eval_matrix[key]['n_kps'])
        table_data.append([key, f'{avg_kps:.0f}', f'{avg_time:.1f}ms', f'{avg_rob:.1f}%'])

    tbl = ax2.table(
        cellText=table_data,
        colLabels=['Metode', 'Keypoints', 'Waktu', 'Avg Precision'],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1F3864')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EBF3FB')
    ax2.set_title('Tabel Perbandingan Metode', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")
    return eval_matrix


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  SISTEM PENCOCOKAN OBJEK BERBASIS FITUR LOKAL")
    print("=" * 60)

    print("\n[1] Membuat dataset sintetis...")
    dataset = create_dataset()
    visualize_dataset(dataset)
    print(f"    {len(OBJECTS)} objek × 5 citra = {len(OBJECTS)*5} total citra")

    print("\n[2] Deteksi & deskripsi fitur...")
    kp_stats = visualize_keypoints(dataset)

    print("\n[3] Feature matching (BF + FLANN + RANSAC)...")
    match_results = visualize_matching(dataset)

    print("\n[4] Bag of Visual Words (k=10,20,50,100)...")
    bovw_results = build_bovw(dataset, vocab_sizes=[10, 20, 50, 100])
    visualize_bovw(bovw_results, dataset)

    print("\n[5] PCA reduksi dimensi...")
    pca_results = pca_analysis(dataset)

    print("\n[6] Evaluasi komprehensif...")
    eval_matrix = comprehensive_evaluation(dataset)

    print("\n[SELESAI] Semua gambar telah disimpan.")

    # Ringkasan
    print("\nRingkasan BoVW:")
    for k, v in bovw_results.items():
        print(f"  k={k:3d}: SVM={v['acc_svm']:.1f}%, kNN={v['acc_knn']:.1f}%, time={v['time_ms']:.0f}ms")

    return dataset, kp_stats, match_results, bovw_results, pca_results, eval_matrix


if __name__ == '__main__':
    main()