from __future__ import annotations
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from nn_RL import neural_multi_scale_optimize

@dataclass
class MultiScaleOptimizer:
	smoothness_factor: float = 0.7
	local_variation_factor: float = 0.3
	window_size: int = 10
	max_iterations_ratio: int = 50
	variation_tolerance_ratio: float = 0.5
	method: str = 'greedy'
	device: str = 'cuda'

	def _prepare_weights(self, weights: Optional[np.ndarray], n_dims: int) -> np.ndarray:
		if weights is None:
			return np.ones(n_dims)
		return np.array(weights)

	def _first_order_cost(self, xi: np.ndarray, xj: np.ndarray, weights: np.ndarray) -> float:
		return float(np.sum(weights * np.abs(xj - xi)))

	def _second_order_cost(self, xa: np.ndarray, xb: np.ndarray, xc: np.ndarray, weights: np.ndarray) -> float:
		second = np.abs(xc - 2.0 * xb + xa)
		return float(np.sum(weights * second))

	def _local_penalty(self, base_cost: float) -> float:
		return base_cost

	def greedy_tsp_construction(self, x: np.ndarray, weights: np.ndarray) -> list:
		n_samples = x.shape[0]
		visited = [False] * n_samples
		path = [0]
		visited[0] = True
		for _ in range(n_samples - 1):
			best_cost = float('inf')
			next_node = -1
			for candidate in range(n_samples):
				if visited[candidate]:
					continue
				if len(path) == 1:
					base_cost = self._first_order_cost(x[path[-1]], x[candidate], weights)
				else:
					base_cost = self._second_order_cost(x[path[-2]], x[path[-1]], x[candidate], weights)
				local_penalty = self._local_penalty(base_cost)
				combined = self.smoothness_factor * base_cost + self.local_variation_factor * local_penalty
				if combined < best_cost:
					best_cost = combined
					next_node = candidate
			if next_node != -1:
				path.append(next_node)
				visited[next_node] = True
		return path

	def calculate_path_cost(self, path: list, x: np.ndarray, weights: np.ndarray) -> float:
		if len(path) < 2:
			return 0.0
		cost = 0.0
		first_cost = self._first_order_cost(x[path[0]], x[path[1]], weights)
		cost += self.smoothness_factor * first_cost + self.local_variation_factor * self._local_penalty(first_cost)
		for i in range(1, len(path) - 1):
			base_cost = self._second_order_cost(x[path[i - 1]], x[path[i]], x[path[i + 1]], weights)
			local_penalty = self._local_penalty(base_cost)
			cost += self.smoothness_factor * base_cost + self.local_variation_factor * local_penalty
		return cost

	def two_opt_improvement(self, path: list, x: np.ndarray, weights: np.ndarray) -> list:
		best_path = path[:]
		best_cost = self.calculate_path_cost(best_path, x, weights)
		improved = True
		iterations = 0
		max_iterations = min(self.max_iterations_ratio, len(path))
		while improved and iterations < max_iterations:
			improved = False
			for i in range(1, len(path) - 2):
				for j in range(i + 1, len(path)):
					if j - i == 1:
						continue
					new_path = path[:i] + path[i:j][::-1] + path[j:]
					new_cost = self.calculate_path_cost(new_path, x, weights)
					if new_cost < best_cost:
						best_path = new_path
						best_cost = new_cost
						improved = True
						path = new_path
						break
				if improved:
					break
			iterations += 1
		return best_path

	def optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
		if self.method == 'neural':
			return self._neural_optimize(x, weights)
		else:
			return self._greedy_optimize(x, weights)

	def _greedy_optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
		n_samples, n_dims = x.shape
		weights_arr = self._prepare_weights(weights, n_dims)
		initial_path = self.greedy_tsp_construction(x, weights_arr)
		optimized_path = self.two_opt_improvement(initial_path, x, weights_arr)
		ranks = np.zeros(n_samples, dtype=int)
		ranks[np.array(optimized_path)] = np.arange(n_samples)
		return ranks.reshape(-1, 1)

	def _neural_optimize(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
		if neural_multi_scale_optimize is None:
			raise RuntimeError("Neural optimization requested but nn_RL module is not available.")
		return neural_multi_scale_optimize(x, weights, n_epochs=50, device=self.device)

	def analyze_metrics(self, data: np.ndarray, ranks: np.ndarray) -> Dict[str, float]:
		ordered_data = data[ranks.flatten()]
		if ordered_data.shape[0] < 3:
			global_diff = 0.0
		else:
			second_diffs = ordered_data[2:] - 2 * ordered_data[1:-1] + ordered_data[:-2]
			global_diff = float(np.mean(np.abs(second_diffs)))
		global_smoothness = 1.0 / (1.0 + global_diff)
		window_size = min(self.window_size, len(ordered_data) // 5)
		local_variations = []
		for i in range(len(ordered_data) - window_size + 1):
			window = ordered_data[i : i + window_size]
			local_std = np.std(window, axis=0)
			local_variations.append(np.mean(local_std))
		avg_local_variation = float(np.mean(local_variations)) if local_variations else 0.0
		trends = []
		for dim in range(data.shape[1]):
			component = ordered_data[:, dim]
			if component.shape[0] < 3:
				second_gradient = np.zeros_like(component)
			else:
				second_gradient = np.gradient(np.gradient(component))
			trend_smoothness = 1.0 / (1.0 + float(np.std(second_gradient)))
			trends.append(trend_smoothness)
		avg_trend_smoothness = float(np.mean(trends)) if trends else 0.0
		balance_score = avg_trend_smoothness * (1.0 + 0.1 * avg_local_variation)
		return {
			'global_smoothness': global_smoothness,
			'local_variation': avg_local_variation,
			'trend_smoothness': avg_trend_smoothness,
			'balance_score': balance_score,
		}


def create_optimizer_configs() -> list:
	configs = [
		{'name': 'Pure Smoothness', 'smoothness_factor': 1.0, 'local_variation_factor': 0.0, 'method': 'greedy'},
		{'name': 'Mostly Smooth', 'smoothness_factor': 0.8, 'local_variation_factor': 0.2, 'method': 'greedy'},
		{'name': 'Balanced', 'smoothness_factor': 0.7, 'local_variation_factor': 0.3, 'method': 'greedy'},
		{'name': 'More Variation', 'smoothness_factor': 0.6, 'local_variation_factor': 0.4, 'method': 'greedy'},
		{'name': 'Neural Balanced', 'smoothness_factor': 0.7, 'local_variation_factor': 0.3, 'method': 'neural'},
	]
	return configs


def multi_scale_optimize(
	x: np.ndarray,
	weights: Optional[np.ndarray] = None,
	smoothness_factor: float = 0.7,
	local_variation_factor: float = 0.3,
	window_size: int = 10,
) -> np.ndarray:
	optimizer = MultiScaleOptimizer(
		smoothness_factor=smoothness_factor,
		local_variation_factor=local_variation_factor,
		window_size=window_size,
	)
	return optimizer.optimize(x, weights)


def analyze_smoothness_and_variation(data: np.ndarray, ranks: np.ndarray) -> Dict[str, float]:
	optimizer = MultiScaleOptimizer()
	return optimizer.analyze_metrics(data, ranks)
