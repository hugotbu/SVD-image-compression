import matplotlib.pyplot as plt
import numpy as np

from src.svd_compressor import manual_svd, matrix_multiply
from src.utils import compression_ratio, load_image_as_matrix, manual_matrix_norm


def create_and_save_summary():
    print("Laster inn bilde og kjører manuell SVD...")

    A = load_image_as_matrix("data/test_image3.jpg")
    m, n = A.shape
    max_k = min(m, n)

    U, S, V_T = manual_svd(A)

    k_low = max(1, int(max_k * 0.05))
    k_med = max(1, int(max_k * 0.15))
    k_high = max(1, int(max_k * 0.50))

    def A_k(k):
        k = min(k, len(S))
        U_k = U[:, :k]
        S_k = S[:k]
        V_T_k = V_T[:k, :]

        U_k_scaled = np.zeros((m, k), dtype=np.float64)
        for i in range(m):
            for j in range(k):
                U_k_scaled[i, j] = U_k[i, j] * S_k[j]

        A_k = matrix_multiply(U_k_scaled, V_T_k)

        for p in range(m):
            for q in range(n):
                if A_k[p, q] < 0.0:
                    A_k[p, q] = 0.0
                elif A_k[p, q] > 255.0:
                    A_k[p, q] = 255.0

        return A_k

    print("Genererer rekonstruksjoner for k_low, k_med og k_high...")
    A_low = A_k(k_low)
    A_med = A_k(k_med)
    A_high = A_k(k_high)

    norm_A = manual_matrix_norm(A)
    err_low = manual_matrix_norm(A - A_low) / norm_A
    err_med = manual_matrix_norm(A - A_med) / norm_A
    err_high = manual_matrix_norm(A - A_high) / norm_A

    rat_low = compression_ratio(A, A_low)
    rat_med = compression_ratio(A, A_med)
    rat_high = compression_ratio(A, A_high)

    print("Bygger visualisering med Matplotlib...")
    fig = plt.figure(figsize=(15, 8))

    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(A, cmap="gray")
    ax1.set_title(f"Originalbilde\n({m}x{n} piksler)")
    ax1.axis("off")

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(A_low, cmap="gray")
    ax2.set_title(f"Lav rang (k={k_low})\nFeil: {err_low:.1%} | {rat_low:.1f}x")
    ax2.axis("off")

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(A_med, cmap="gray")
    ax3.set_title(f"Middels rang (k={k_med})\nFeil: {err_med:.1%} | {rat_med:.1f}x")
    ax3.axis("off")

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(A_high, cmap="gray")
    ax4.set_title(f"Høy rang (k={k_high})\nFeil: {err_high:.1%} | {rat_high:.1f}x")
    ax4.axis("off")

    ax5 = plt.subplot(2, 2, 3)
    ax5.plot(range(1, len(S) + 1), S, "b-o", linewidth=2, markersize=3)
    ax5.set_yscale("log")
    ax5.set_title("Singulærverdispekter (Logaritmisk skala)")
    ax5.set_xlabel("Singulærverdi-indeks (i)")
    ax5.set_ylabel("Singulærverdi (σ_i)")
    ax5.grid(True, which="both", linestyle="--", alpha=0.5)

    step = max(1, max_k // 20)
    k_samples = list(range(1, max_k//2, step))
    errors = [manual_matrix_norm(A - A_k(k_v)) / norm_A for k_v in k_samples]

    ax6 = plt.subplot(2, 2, 4)
    ax6.plot(k_samples, errors, "r-s", linewidth=2, markersize=4)
    ax6.set_title("Relativ feil vs. rang k")
    ax6.set_xlabel("Rang k")
    ax6.set_ylabel("Relativ feil")
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = "data/svd_summary_result.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Suksess! Oppsummeringsbilde ble lagret i: {output_path}")


if __name__ == "__main__":
    create_and_save_summary()