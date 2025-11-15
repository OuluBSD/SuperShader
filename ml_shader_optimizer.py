"""
ML Training for Shader Optimization System
Part of SuperShader Project - Phase 7: Neural Network Integration and AI Features

This module implements machine learning systems for optimizing shader parameters,
predicting optimal shader configurations, and using reinforcement learning for
dynamic shader optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import json
import random


@dataclass
class ShaderConfig:
    """Represents a shader configuration with parameters"""
    name: str
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float]  # fps, memory usage, etc.
    quality_metrics: Dict[str, float]      # visual quality scores
    hardware_target: str                   # GPU model, platform, etc.


class ShaderParameterOptimizer:
    """
    System for training models to optimize shader parameters
    """
    
    def __init__(self):
        self.training_data: List[ShaderConfig] = []
        self.optimization_models: Dict[str, Any] = {}
        self.current_config: Optional[ShaderConfig] = None
    
    def add_training_sample(self, config: ShaderConfig) -> None:
        """Add a training sample to the optimization system"""
        self.training_data.append(config)
    
    def train_optimization_model(self, target_metric: str = "performance") -> None:
        """
        Train a model to optimize shader configurations based on performance/quality
        Note: This is a simplified implementation; a real system would use more complex ML
        """
        if not self.training_data:
            print("No training data available")
            return
        
        # In a real implementation, this would train a neural network
        # For now, we'll simulate training by finding optimal parameter ranges
        
        param_ranges: Dict[str, Tuple[float, float]] = {}
        
        for config in self.training_data:
            for param_name, param_value in config.parameters.items():
                if param_name not in param_ranges:
                    param_ranges[param_name] = (param_value, param_value)
                else:
                    current_min, current_max = param_ranges[param_name]
                    param_ranges[param_name] = (
                        min(current_min, param_value),
                        max(current_max, param_value)
                    )
        
        # Store the model (in this case, the parameter ranges)
        self.optimization_models[target_metric] = param_ranges
    
    def predict_optimal_parameters(self, target_metric: str = "performance", 
                                 constraints: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """
        Predict optimal shader parameters for the given target metric
        """
        if target_metric not in self.optimization_models:
            print(f"No model trained for {target_metric}")
            return {}
        
        param_ranges = self.optimization_models[target_metric]
        optimal_params = {}
        
        for param_name, (min_val, max_val) in param_ranges.items():
            # Apply constraints if provided
            if constraints and param_name in constraints:
                min_val = max(min_val, constraints[param_name][0])
                max_val = min(max_val, constraints[param_name][1])
            
            # For this simple model, pick the midpoint (could be more sophisticated)
            optimal_params[param_name] = (min_val + max_val) / 2.0
        
        return optimal_params
    
    def optimize_shader_config(self, base_config: ShaderConfig, 
                             target_metric: str = "performance") -> ShaderConfig:
        """
        Optimize a shader configuration based on trained models
        """
        optimal_params = self.predict_optimal_parameters(target_metric)
        
        # Create new config with optimized parameters
        new_config = ShaderConfig(
            name=base_config.name + "_optimized",
            parameters=optimal_params,
            performance_metrics={},
            quality_metrics={},
            hardware_target=base_config.hardware_target
        )
        
        self.current_config = new_config
        return new_config


class ShaderConfigurationSelector:
    """
    Learning-based shader selection system
    """
    
    def __init__(self):
        self.config_performance_history: Dict[str, List[float]] = {}
        self.config_quality_history: Dict[str, List[float]] = {}
        self.selection_model: Dict[str, Any] = {}
    
    def record_config_performance(self, config_name: str, performance: float) -> None:
        """Record performance data for a shader configuration"""
        if config_name not in self.config_performance_history:
            self.config_performance_history[config_name] = []
        self.config_performance_history[config_name].append(performance)
    
    def record_config_quality(self, config_name: str, quality: float) -> None:
        """Record quality data for a shader configuration"""
        if config_name not in self.config_quality_history:
            self.config_quality_history[config_name] = []
        self.config_quality_history[config_name].append(quality)
    
    def predict_config_performance(self, config_name: str) -> float:
        """Predict performance for a configuration based on history"""
        if config_name in self.config_performance_history:
            history = self.config_performance_history[config_name]
            if history:
                # Simple average prediction (could be more sophisticated)
                return sum(history) / len(history)
        return 0.0
    
    def select_best_config(self, available_configs: List[ShaderConfig], 
                          performance_weight: float = 0.5) -> Optional[ShaderConfig]:
        """
        Select the best configuration based on learned preferences
        """
        if not available_configs:
            return None
        
        best_config = None
        best_score = float('-inf')
        
        for config in available_configs:
            perf_score = self.predict_config_performance(config.name)
            # Quality score could be similarly predicted from history
            
            # Combined score with given weights
            combined_score = performance_weight * perf_score + (1 - performance_weight) * 0.5  # placeholder for quality
            
            if combined_score > best_score:
                best_score = combined_score
                best_config = config
        
        return best_config


class ReinforcementLearningOptimizer:
    """
    Reinforcement learning system for dynamic shader optimization
    """
    
    def __init__(self, action_space: List[str], state_space: List[str]):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table: Dict[Tuple[str, str], float] = {}  # state-action -> value
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
    
    def get_state_action_key(self, state: str, action: str) -> Tuple[str, str]:
        """Create a key for the Q-table"""
        return (state, action)
    
    def get_q_value(self, state: str, action: str) -> float:
        """Get the Q-value for a state-action pair"""
        key = self.get_state_action_key(state, action)
        return self.q_table.get(key, 0.0)
    
    def set_q_value(self, state: str, action: str, value: float) -> None:
        """Set the Q-value for a state-action pair"""
        key = self.get_state_action_key(state, action)
        self.q_table[key] = value
    
    def choose_action(self, state: str) -> str:
        """
        Choose an action based on current Q-values and exploration policy
        """
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.choice(self.action_space)
        else:
            # Exploit: choose best known action
            q_values = {action: self.get_q_value(state, action) for action in self.action_space}
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str) -> None:
        """
        Update Q-value based on experience
        """
        current_q = self.get_q_value(state, action)
        
        # Calculate max Q-value for next state
        next_q_values = [self.get_q_value(next_state, a) for a in self.action_space]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.set_q_value(state, action, new_q)
    
    def train_step(self, state: str, action: str, reward: float, next_state: str) -> None:
        """
        Perform one step of training
        """
        self.update_q_value(state, action, reward, next_state)


class MLShaderOptimizerSystem:
    """
    Main system for ML-based shader optimization
    """
    
    def __init__(self):
        self.parameter_optimizer = ShaderParameterOptimizer()
        self.config_selector = ShaderConfigurationSelector()
        self.rl_optimizer = ReinforcementLearningOptimizer(
            action_space=["increase_quality", "decrease_quality", "increase_performance", "decrease_performance"],
            state_space=["low_fps", "medium_fps", "high_fps", "low_quality", "medium_quality", "high_quality"]
        )
        self.optimization_history: List[Dict[str, Any]] = []
    
    def add_training_data(self, config: ShaderConfig) -> None:
        """Add training data to the system"""
        self.parameter_optimizer.add_training_sample(config)
        self.config_selector.record_config_performance(config.name, 
                                                      config.performance_metrics.get('fps', 0))
        self.config_selector.record_config_quality(config.name, 
                                                  config.quality_metrics.get('score', 0))
    
    def optimize_parameters(self, base_config: ShaderConfig, target: str = "performance") -> ShaderConfig:
        """Optimize shader parameters using ML techniques"""
        # Train model if not already done
        if not self.parameter_optimizer.optimization_models:
            self.parameter_optimizer.train_optimization_model(target)
        
        # Perform optimization
        return self.parameter_optimizer.optimize_shader_config(base_config, target)
    
    def select_best_configuration(self, available_configs: List[ShaderConfig]) -> Optional[ShaderConfig]:
        """Select the best configuration from available options"""
        return self.config_selector.select_best_config(available_configs)
    
    def get_rl_suggestion(self, current_state: Dict[str, float]) -> str:
        """
        Get a reinforcement learning suggestion for optimization
        """
        # Convert current state to RL state
        fps = current_state.get('fps', 30)
        quality = current_state.get('quality', 0.5)
        
        if fps < 30:
            rl_state = "low_fps"
        elif fps < 60:
            rl_state = "medium_fps"
        else:
            rl_state = "high_fps"
        
        return self.rl_optimizer.choose_action(rl_state)
    
    def log_optimization_result(self, result: Dict[str, Any]) -> None:
        """Log an optimization result for historical analysis"""
        self.optimization_history.append(result)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a report on optimization results"""
        if not self.optimization_history:
            return {"message": "No optimization data available"}
        
        total_improvements = len(self.optimization_history)
        avg_performance_gain = np.mean([
            result.get('performance_gain', 0) for result in self.optimization_history
            if 'performance_gain' in result
        ]) if any('performance_gain' in result for result in self.optimization_history) else 0
        
        return {
            "total_optimizations": total_improvements,
            "average_performance_gain": avg_performance_gain,
            "recent_results": self.optimization_history[-5:]  # Last 5 results
        }


def main():
    """
    Example usage of the ML Shader Optimization System
    """
    print("ML Training for Shader Optimization System")
    print("Part of SuperShader Project - Phase 7")
    
    # Create optimization system
    optimizer = MLShaderOptimizerSystem()
    
    # Create some sample training data
    sample_configs = [
        ShaderConfig(
            name="config_1",
            parameters={"detail_level": 0.5, "shadow_quality": 0.8, "aa_level": 2},
            performance_metrics={"fps": 45.2, "memory_mb": 1200},
            quality_metrics={"score": 0.75},
            hardware_target="rtx_3080"
        ),
        ShaderConfig(
            name="config_2", 
            parameters={"detail_level": 0.8, "shadow_quality": 0.9, "aa_level": 4},
            performance_metrics={"fps": 28.3, "memory_mb": 1800},
            quality_metrics={"score": 0.92},
            hardware_target="rtx_3080"
        ),
        ShaderConfig(
            name="config_3",
            parameters={"detail_level": 0.3, "shadow_quality": 0.5, "aa_level": 1},
            performance_metrics={"fps": 60.0, "memory_mb": 800},
            quality_metrics={"score": 0.45},
            hardware_target="rtx_3080"
        )
    ]
    
    # Add training data
    for config in sample_configs:
        optimizer.add_training_data(config)
    
    # Create a base configuration to optimize
    base_config = ShaderConfig(
        name="default_config",
        parameters={"detail_level": 0.6, "shadow_quality": 0.7, "aa_level": 2},
        performance_metrics={"fps": 40.0, "memory_mb": 1000},
        quality_metrics={"score": 0.65},
        hardware_target="rtx_3080"
    )
    
    # Optimize parameters
    optimized_config = optimizer.optimize_parameters(base_config, "performance")
    print(f"Optimized config: {optimized_config.name}")
    print(f"Parameters: {optimized_config.parameters}")
    
    # Get RL suggestion
    current_state = {"fps": 30.0, "quality": 0.7}
    rl_suggestion = optimizer.get_rl_suggestion(current_state)
    print(f"RL Suggestion for state {current_state}: {rl_suggestion}")
    
    # Generate report
    report = optimizer.get_optimization_report()
    print("Optimization Report:", report)


if __name__ == "__main__":
    main()