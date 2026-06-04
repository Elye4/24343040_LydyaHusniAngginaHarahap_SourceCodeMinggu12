import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt

def latihan_1():

    print("COMPARISON OF LOCAL FEATURE DESCRIPTORS")
    print("=" * 50)

    # ==================================================
    # MEMBUAT CITRA UJI
    # ==================================================
    def create_test_images():

        # Citra original
        img1 = np.zeros((300, 400), dtype=np.uint8)

        # Tambahkan objek
        cv2.rectangle(img1, (50, 50), (150, 150), 220, -1)
        cv2.circle(img1, (250, 100), 50, 180, -1)
        cv2.rectangle(img1, (100, 200), (220, 260), 120, -1)
        cv2.line(img1, (320, 50), (360, 200), 255, 4)

        # ---------------------------
        # Transformasi
        # ---------------------------

        # Rotasi
        M = cv2.getRotationMatrix2D((200, 150), 25, 1.0)
        img2 = cv2.warpAffine(img1, M, (400, 300))

        # Brightness
        img2 = cv2.convertScaleAbs(img2, alpha=1.1, beta=10)

        # Noise yang benar
        noise = np.random.normal(0, 8, img2.shape)

        img2 = img2.astype(np.float32)
        img2 = img2 + noise

        img2 = np.clip(img2, 0, 255).astype(np.uint8)

        return img1, img2

    img1, img2 = create_test_images()

    # ==================================================
    # FEATURE DETECTORS
    # ==================================================

    # SIFT
    sift = cv2.SIFT_create(nfeatures=120)

    # ORB
    orb = cv2.ORB_create(nfeatures=120)

    # SURF (opsional)
    surf = None

    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    except:
        print("SURF tidak tersedia di OpenCV ini")

    # ==================================================
    # EKSTRAKSI FEATURE
    # ==================================================
    def extract_features(detector, image):

        if detector is None:
            return None, None, None

        keypoints, descriptors = detector.detectAndCompute(image, None)

        img_kp = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        return keypoints, descriptors, img_kp

    # SIFT
    kp1_sift, desc1_sift, img1_sift = extract_features(sift, img1)
    kp2_sift, desc2_sift, img2_sift = extract_features(sift, img2)

    # SURF
    if surf is not None:
        kp1_surf, desc1_surf, img1_surf = extract_features(surf, img1)
        kp2_surf, desc2_surf, img2_surf = extract_features(surf, img2)

    # ORB
    kp1_orb, desc1_orb, img1_orb = extract_features(orb, img1)
    kp2_orb, desc2_orb, img2_orb = extract_features(orb, img2)

    # ==================================================
    # INFORMASI HASIL
    # ==================================================
    print("\nFEATURE EXTRACTION RESULTS")
    print("-" * 50)

    print(f"SIFT Image1 Keypoints : {len(kp1_sift)}")
    print(f"SIFT Image2 Keypoints : {len(kp2_sift)}")

    if surf is not None:
        print(f"SURF Image1 Keypoints : {len(kp1_surf)}")
        print(f"SURF Image2 Keypoints : {len(kp2_surf)}")

    print(f"ORB Image1 Keypoints  : {len(kp1_orb)}")
    print(f"ORB Image2 Keypoints  : {len(kp2_orb)}")

    # ==================================================
    # VISUALISASI FEATURE
    # ==================================================

    rows = 3 if surf is not None else 2

    fig, axes = plt.subplots(rows, 3, figsize=(14, 10))

    # -------------------------
    # SIFT
    # -------------------------
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img1_sift, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"SIFT Keypoints\n{len(kp1_sift)} points")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(img2_sift, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"SIFT Transformed\n{len(kp2_sift)} points")
    axes[0, 2].axis("off")

    # -------------------------
    # SURF
    # -------------------------
    if surf is not None:

        axes[1, 0].imshow(img1, cmap='gray')
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(img1_surf, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"SURF Keypoints\n{len(kp1_surf)} points")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(cv2.cvtColor(img2_surf, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f"SURF Transformed\n{len(kp2_surf)} points")
        axes[1, 2].axis("off")

        orb_row = 2

    else:
        orb_row = 1

    # -------------------------
    # ORB
    # -------------------------
    axes[orb_row, 0].imshow(img1, cmap='gray')
    axes[orb_row, 0].set_title("Original Image")
    axes[orb_row, 0].axis("off")

    axes[orb_row, 1].imshow(cv2.cvtColor(img1_orb, cv2.COLOR_BGR2RGB))
    axes[orb_row, 1].set_title(f"ORB Keypoints\n{len(kp1_orb)} points")
    axes[orb_row, 1].axis("off")

    axes[orb_row, 2].imshow(cv2.cvtColor(img2_orb, cv2.COLOR_BGR2RGB))
    axes[orb_row, 2].set_title(f"ORB Transformed\n{len(kp2_orb)} points")
    axes[orb_row, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # ==================================================
    # FEATURE MATCHING
    # ==================================================
    print("\nFEATURE MATCHING RESULTS")
    print("-" * 50)

    def feature_matching(desc1, desc2, method):

        if method == "ORB":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = matcher.match(desc1, desc2)

        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    # SIFT matching
    sift_matches = feature_matching(desc1_sift, desc2_sift, "SIFT")
    print(f"SIFT Matches : {len(sift_matches)}")

    # SURF matching
    if surf is not None:
        surf_matches = feature_matching(desc1_surf, desc2_surf, "SURF")
        print(f"SURF Matches : {len(surf_matches)}")

    # ORB matching
    orb_matches = feature_matching(desc1_orb, desc2_orb, "ORB")
    print(f"ORB Matches  : {len(orb_matches)}")

    # ==================================================
    # VISUALISASI MATCHING
    # ==================================================

    plt.figure(figsize=(15, 10))

    # SIFT
    sift_match_img = cv2.drawMatches(
        img1,
        kp1_sift,
        img2,
        kp2_sift,
        sift_matches[:20],
        None,
        flags=2
    )

    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(sift_match_img, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Feature Matching")
    plt.axis("off")

    # ORB
    orb_match_img = cv2.drawMatches(
        img1,
        kp1_orb,
        img2,
        kp2_orb,
        orb_matches[:20],
        None,
        flags=2
    )

    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(orb_match_img, cv2.COLOR_BGR2RGB))
    plt.title("ORB Feature Matching")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==================================================
# JALANKAN PROGRAM
# ==================================================

latihan_1()