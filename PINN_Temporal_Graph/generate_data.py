"""
Generate synthetic eigengene data for testing PINN Temporal Graph framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_circadian_eigengenes(n_samples=100, n_eigengenes=5, period=24.0, noise_level=0.1):
    """
    Generate synthetic eigengene data mimicking PCA results from circadian expression data

    Key characteristics:
    - All eigengenes share the SAME fundamental frequency (24-hour circadian rhythm)
    - Different eigengenes have different PHASE shifts (0°, 72°, 144°, 216°, 288° for 5 eigengenes)
    - Different amplitudes (first PC has largest variance)
    - Optional harmonics for biological realism

    Args:
        n_samples: Number of time points/samples
        n_eigengenes: Number of eigengenes (principal components)
        period: Circadian period in hours (same for all eigengenes)
        noise_level: Amount of noise to add

    Returns:
        time_points: Array of time values
        eigengenes: (n_samples, n_eigengenes) array of eigengene expressions
    """
    # Time points (assuming uniform sampling over the period)
    time_points = np.linspace(0, period, n_samples)

    # Initialize eigengenes array
    eigengenes = np.zeros((n_samples, n_eigengenes))

    # Generate each eigengene with different characteristics
    for i in range(n_eigengenes):
        # All eigengenes share the SAME fundamental frequency (circadian rhythm)
        freq = 2 * np.pi / period

        # Different phase shifts for each eigengene (0, 2π/n, 4π/n, etc.)
        phase_shift = i * 2 * np.pi / n_eigengenes

        # Different amplitudes (first PC usually has largest variance)
        amplitude = 2.0 / (i + 1)  # Decreasing amplitude

        # All eigengenes have the same base frequency but different phases
        signal = amplitude * np.sin(freq * time_points + phase_shift)

        # Add second harmonic for more realistic circadian patterns (same frequency for all)
        if i < 3:  # Only for first few PCs
            signal += 0.3 * amplitude * np.sin(2 * freq * time_points + phase_shift)

        # Add noise
        noise = noise_level * np.random.randn(n_samples)
        signal += noise

        eigengenes[:, i] = signal

    # Z-score normalize (typical for PCA results)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    eigengenes_normalized = scaler.fit_transform(eigengenes)

    return time_points, eigengenes_normalized

def add_biological_realism(eigengenes, time_points):
    """
    Add biological realism to synthetic eigengenes
    """
    n_samples, n_eigengenes = eigengenes.shape

    # Add slight damping (circadian amplitude decreases over time in some conditions)
    damping_factor = np.exp(-time_points / (24 * 7))  # Week-long experiment
    eigengenes = eigengenes * damping_factor[:, np.newaxis]

    # Add batch effects (simulating different experimental days)
    batch_size = 20  # 20 samples per batch
    n_batches = n_samples // batch_size

    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        # Small batch-specific offset
        batch_offset = 0.1 * np.random.randn(n_eigengenes)
        eigengenes[start_idx:end_idx] += batch_offset

    return eigengenes

def save_eigengene_data(time_points, eigengenes, output_path='synthetic_eigengenes.csv'):
    """
    Save eigengene data to CSV file
    """
    # Create DataFrame
    df = pd.DataFrame(eigengenes, columns=[f'Eigengene_{i+1}' for i in range(eigengenes.shape[1])])
    df.insert(0, 'Time', time_points)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic eigengene data saved to {output_path}")
    print(f"Shape: {eigengenes.shape} (samples x eigengenes)")
    print(f"Time range: {time_points[0]:.1f} - {time_points[-1]:.1f} hours")

    return df

def plot_eigengenes(time_points, eigengenes, save_path=None):
    """
    Plot the generated eigengene time series
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(5, eigengenes.shape[1])):
        ax = axes[i]
        ax.plot(time_points, eigengenes[:, i], 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(f'Eigengene {i+1}')
        ax.set_title(f'Eigengene {i+1}')
        ax.grid(True, alpha=0.3)

        # Add period markers
        for period_start in range(0, int(time_points[-1]), 24):
            ax.axvline(x=period_start, color='r', linestyle='--', alpha=0.5)

    # Plot correlation matrix
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
    """
    Analyze properties of generated eigengenes
    """
    n_eigengenes = eigengenes.shape[1]  # Define n_eigengenes here

    print("\nEigengene Analysis:")
    print("=" * 50)

    # Basic statistics
    print(f"Number of samples: {eigengenes.shape[0]}")
    print(f"Number of eigengenes: {eigengenes.shape[1]}")
    print(f"Time range: {time_points[0]:.2f} - {time_points[-1]:.2f} hours")
    print(f"Mean expression: {np.mean(eigengenes):.2f} ± {np.std(eigengenes):.2f}")

    # Variance explained (simulating PCA)
    variances = np.var(eigengenes, axis=0)
    total_variance = np.sum(variances)
    explained_variance_ratio = variances / total_variance

    print("\nVariance explained by each eigengene:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"  Eigengene {i+1}: {ratio:.1%}")

    # Periodicity check using FFT
    from scipy import signal

    print("\nPeriodicity analysis (all eigengenes should have ~24h period):")
    for i in range(min(3, eigengenes.shape[1])):
        # Simple periodogram
        eigengene = eigengenes[:, i]
        freqs = np.fft.fftfreq(len(time_points), d=(time_points[1] - time_points[0]))
        power = np.abs(np.fft.fft(eigengene)) ** 2

        # Find dominant frequency
        valid_freqs = freqs > 0
        dominant_freq_idx = np.argmax(power[valid_freqs])
        dominant_freq = freqs[valid_freqs][dominant_freq_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0

        print(f"  Eigengene {i+1}: dominant period = {dominant_period:.1f} hours (phase shift: {i*360/n_eigengenes:.0f}°)")

    # Autocorrelation
    print("\nAutocorrelation at 24h lag:")
    for i in range(min(3, eigengenes.shape[1])):
        eigengene = eigengenes[:, i]
        autocorr = np.correlate(eigengene, eigengene, mode='full')
        lag_24_samples = int(24 / (time_points[1] - time_points[0]))
        if lag_24_samples < len(autocorr) // 2:
            autocorr_24h = autocorr[len(autocorr)//2 + lag_24_samples]
            print(f"  Eigengene {i+1}: autocorrelation = {autocorr_24h:.3f}")
def main():
    """
    Generate synthetic eigengene data
    """
    # Parameters
    n_samples = 100
    n_eigengenes = 5
    period = 24.0
    noise_level = 0.1

    print("Generating synthetic eigengene data...")
    print(f"Samples: {n_samples}, Eigengenes: {n_eigengenes}, Period: {period}h")

    # Generate basic circadian patterns
    time_points, eigengenes = generate_circadian_eigengenes(
        n_samples=n_samples,
        n_eigengenes=n_eigengenes,
        period=period,
        noise_level=noise_level
    )

    # Add biological realism
    eigengenes = add_biological_realism(eigengenes, time_points)

    # Analyze properties
    analyze_eigengene_properties(time_points, eigengenes)

    # Save data
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'synthetic_eigengenes_100x5.csv'
    plot_path = output_dir / 'synthetic_eigengenes_plot.png'

    df = save_eigengene_data(time_points, eigengenes, csv_path)

    # Plot results
    plot_eigengenes(time_points, eigengenes, plot_path)

    print(f"\nData preview:")
    print(df.head())
    print("\nData generation completed!")

if __name__ == "__main__":
    main()