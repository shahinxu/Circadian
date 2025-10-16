import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_circadian_eigengenes(n_samples=100, n_eigengenes=5, period=24.0, noise_level=0.1):
    time_points = np.linspace(0, period, n_samples)

    eigengenes = np.zeros((n_samples, n_eigengenes))

    for i in range(n_eigengenes):
        freq = 2 * np.pi / period

        phase_shift = i * 2 * np.pi / n_eigengenes

        amplitude = 2.0 / (i + 1)

        signal = amplitude * np.sin(freq * time_points + phase_shift)

        if i < 3:
            signal += 0.3 * amplitude * np.sin(2 * freq * time_points + phase_shift)

        noise = noise_level * np.random.randn(n_samples)
        signal += noise

        eigengenes[:, i] = signal

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    eigengenes_normalized = scaler.fit_transform(eigengenes)

    return time_points, eigengenes_normalized

def add_biological_realism(eigengenes, time_points):
    n_samples, n_eigengenes = eigengenes.shape

    damping_factor = np.exp(-time_points / (24 * 7))
    eigengenes = eigengenes * damping_factor[:, np.newaxis]

    batch_size = 20
    n_batches = n_samples // batch_size

    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        batch_offset = 0.1 * np.random.randn(n_eigengenes)
        eigengenes[start_idx:end_idx] += batch_offset

    return eigengenes

def save_eigengene_data(time_points, eigengenes, output_path='synthetic_eigengenes.csv'):
    df = pd.DataFrame(eigengenes, columns=[f'Eigengene_{i+1}' for i in range(eigengenes.shape[1])])
    df.insert(0, 'Time', time_points)

    df.to_csv(output_path, index=False)
    print(f"Synthetic eigengene data saved to {output_path}")
    print(f"Shape: {eigengenes.shape} (samples x eigengenes)")
    print(f"Time range: {time_points[0]:.1f} - {time_points[-1]:.1f} hours")

    return df

def plot_eigengenes(time_points, eigengenes, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(5, eigengenes.shape[1])):
        ax = axes[i]
        ax.plot(time_points, eigengenes[:, i], 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(f'Eigengene {i+1}')
        ax.set_title(f'Eigengene {i+1}')
        ax.grid(True, alpha=0.3)

        for period_start in range(0, int(time_points[-1]), 24):
            ax.axvline(x=period_start, color='r', linestyle='--', alpha=0.5)

    if eigengenes.shape[1] > 1:
        ax = axes[5]
        corr_matrix = np.corrcoef(eigengenes.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Eigengene Correlations')
        ax.set_xlabel('Eigengene')
        ax.set_ylabel('Eigengene')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def analyze_eigengene_properties(time_points, eigengenes):
    n_eigengenes = eigengenes.shape[1]

    print("\nEigengene Analysis:")
    print("=" * 50)

    print(f"Number of samples: {eigengenes.shape[0]}")
    print(f"Number of eigengenes: {eigengenes.shape[1]}")
    print(f"Time range: {time_points[0]:.2f} - {time_points[-1]:.2f} hours")
    print(f"Mean expression: {np.mean(eigengenes):.2f} ± {np.std(eigengenes):.2f}")

    variances = np.var(eigengenes, axis=0)
    total_variance = np.sum(variances)
    explained_variance_ratio = variances / total_variance

    print("\nVariance explained by each eigengene:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"  Eigengene {i+1}: {ratio:.1%}")

    print("\nPeriodicity analysis (all eigengenes should have ~24h period):")
    for i in range(min(3, eigengenes.shape[1])):
        eigengene = eigengenes[:, i]
        freqs = np.fft.fftfreq(len(time_points), d=(time_points[1] - time_points[0]))
        power = np.abs(np.fft.fft(eigengene)) ** 2

        valid_freqs = freqs > 0
        dominant_freq_idx = np.argmax(power[valid_freqs])
        dominant_freq = freqs[valid_freqs][dominant_freq_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0

        print(f"  Eigengene {i+1}: dominant period = {dominant_period:.1f} hours (phase shift: {i*360/n_eigengenes:.0f}°)")

    print("\nAutocorrelation at 24h lag:")
    for i in range(min(3, eigengenes.shape[1])):
        eigengene = eigengenes[:, i]
        autocorr = np.correlate(eigengene, eigengene, mode='full')
        lag_24_samples = int(24 / (time_points[1] - time_points[0]))
        if lag_24_samples < len(autocorr) // 2:
            autocorr_24h = autocorr[len(autocorr)//2 + lag_24_samples]
            print(f"  Eigengene {i+1}: autocorrelation = {autocorr_24h:.3f}")

def main():
    n_samples = 100
    n_eigengenes = 5
    period = 24.0
    noise_level = 0.1

    print("Generating synthetic eigengene data...")
    print(f"Samples: {n_samples}, Eigengenes: {n_eigengenes}, Period: {period}h")

    time_points, eigengenes = generate_circadian_eigengenes(
        n_samples=n_samples,
        n_eigengenes=n_eigengenes,
        period=period,
        noise_level=noise_level
    )

    eigengenes = add_biological_realism(eigengenes, time_points)

    analyze_eigengene_properties(time_points, eigengenes)

    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'synthetic_eigengenes_100x5.csv'
    plot_path = output_dir / 'synthetic_eigengenes_plot.png'

    df = save_eigengene_data(time_points, eigengenes, csv_path)

    plot_eigengenes(time_points, eigengenes, plot_path)

    print(f"\nData preview:")
    print(df.head())
    print("\nData generation completed!")

if __name__ == "__main__":
    main()