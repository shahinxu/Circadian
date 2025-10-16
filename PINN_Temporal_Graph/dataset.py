import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CircadianDataset:
    def __init__(self, expression_data, time_points, normalize=True):
        self.time_points = torch.tensor(time_points, dtype=torch.float32)
        self.expressions = torch.tensor(expression_data, dtype=torch.float32)

        if normalize:
            scaler = StandardScaler()
            self.expressions = torch.tensor(
                scaler.fit_transform(self.expressions.numpy()),
                dtype=torch.float32
            )

    def __len__(self):
        return len(self.time_points)

    def __getitem__(self, idx):
        return self.time_points[idx], self.expressions[idx]

def load_expression_data(file_path, time_col='time', sep='\t'):
    df = pd.read_csv(file_path, sep=sep, index_col=0)

    if time_col in df.columns:
        time_points = df[time_col].values
        expression_data = df.drop(columns=[time_col]).values
    else:
        time_points = df.index.values if df.index.name == time_col else df.iloc[:, 0].values
        expression_data = df.iloc[:, 1:].values

    return expression_data, time_points

def create_time_points(n_points=24, period=24.0):
    return torch.linspace(0, period, n_points)

def interpolate_to_grid(expression_data, time_points, target_times):
    from scipy import interpolate

    interpolated = []
    for gene_idx in range(expression_data.shape[1]):
        gene_expr = expression_data[:, gene_idx]
        f = interpolate.interp1d(time_points, gene_expr, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        interpolated.append(f(target_times))

    return torch.tensor(np.stack(interpolated, axis=1), dtype=torch.float32)

def detect_periodic_patterns(expression_data, time_points, min_period=20, max_period=28):
    from scipy import signal

    periods = []
    for gene_idx in range(expression_data.shape[1]):
        gene_expr = expression_data[:, gene_idx]

        gene_expr_detrended = signal.detrend(gene_expr)

        fft = np.fft.fft(gene_expr_detrended)
        freqs = np.fft.fftfreq(len(time_points), d=(time_points[1] - time_points[0]))

        power = np.abs(fft) ** 2
        peaks, _ = signal.find_peaks(power[freqs > 0], height=np.max(power) * 0.1)

        if len(peaks) > 0:
            dominant_freq = freqs[freqs > 0][peaks[0]]
            period = 1.0 / dominant_freq if dominant_freq > 0 else 0
            if min_period <= period <= max_period:
                periods.append(period)

    return np.mean(periods) if periods else 24.0

def compute_eigengenes(expression_data, n_components=5):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    eigengenes = pca.fit_transform(expression_data)

    return torch.tensor(eigengenes, dtype=torch.float32), pca

def prepare_training_data(data_path, n_eigengenes=5, n_time_points=100):
    expression_data, time_points = load_expression_data(data_path)

    detected_period = detect_periodic_patterns(expression_data, time_points)
    print(f"Detected circadian period: {detected_period:.1f} hours")

    target_times = create_time_points(n_time_points, detected_period)
    expression_interp = interpolate_to_grid(expression_data, time_points, target_times.numpy())

    eigengenes, pca = compute_eigengenes(expression_interp.numpy(), n_eigengenes)

    print(f"Data prepared: {eigengenes.shape[0]} time points, {eigengenes.shape[1]} eigengenes")

    return eigengenes, target_times

def create_graph_from_eigengenes(eigengenes, method='correlation'):
    if method == 'correlation':
        corr_matrix = torch.corrcoef(eigengenes.T)
        adjacency = torch.abs(corr_matrix)

        threshold = torch.quantile(adjacency, 0.8)
        adjacency = (adjacency > threshold).float()

        adjacency.fill_diagonal_(0)

    elif method == 'mutual_info':
        adjacency = torch.rand(eigengenes.shape[1], eigengenes.shape[1])
        adjacency = (adjacency + adjacency.T) / 2
        adjacency.fill_diagonal_(0)

    return adjacency

def validate_circadian_data(expression_data, time_points):
    issues = []

    if torch.isnan(expression_data).any():
        issues.append("Data contains NaN values")

    time_diffs = torch.diff(time_points)
    if torch.std(time_diffs) > torch.mean(time_diffs) * 0.5:
        issues.append("Irregular time point spacing")

    period = detect_periodic_patterns(expression_data.numpy(), time_points.numpy())
    if not (20 <= period <= 28):
        issues.append(f"Suspicious period detected: {period:.1f} hours")

    return issues