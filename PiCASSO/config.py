class Config:
    """Shared configuration for PiCASSO pipeline (minimal set)."""
    # PCA components
    N_COMPONENTS = 5
    
    # Optimizer settings
    DEFAULT_WINDOW_SIZE = 10
    MAX_ITERATIONS_RATIO = 5
    VARIATION_TOLERANCE_RATIO = 0.5

    # Default data paths
    DEFAULT_EXPRESSION_FILE = "../data/GSE146773/expression.csv"
    DEFAULT_METADATA_FILE = "../data/GSE146773/metadata.csv"

    @staticmethod
    def get_eigengene_weights(n_components: int):
        """Generate eigengene weights based on component importance."""
        if n_components <= 50:
            weights = []
            weights.extend([1.0])
            weights.extend([0.8] * min(10, n_components - 1))
            if n_components > 11:
                weights.extend([0.6] * min(15, n_components - 11))
            if n_components > 26:
                weights.extend([0.4] * (n_components - 26))
            return weights[:n_components]
        else:
            weights = []
            weights.extend([1.0])
            weights.extend([0.8] * 10)
            weights.extend([0.6] * 15)
            weights.extend([0.4] * 24)
            remaining = n_components - 50
            if remaining > 0:
                weights.extend([0.2] * min(25, remaining))
                if remaining > 25:
                    weights.extend([0.1] * (remaining - 25))
            return weights[:n_components]
