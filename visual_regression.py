"""
Visual Regression Testing System
Part of SuperShader Project - Phase 8: Testing and Quality Assurance

This module implements image-based comparison for shader output,
creates a test framework for visual verification, develops tools
for detecting visual artifacts, and creates automated screenshot
and comparison systems.
"""

import numpy as np
from PIL import Image, ImageChops, ImageStat
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path


@dataclass
class VisualTestResult:
    """Represents the result of a visual regression test"""
    test_id: str
    shader_name: str
    baseline_image_path: str
    current_image_path: str
    passed: bool
    similarity_score: float
    pixel_difference_count: int
    max_difference: float
    mean_difference: float
    execution_time: float
    artifacts_detected: List[str]


class ImageComparator:
    """
    System for comparing rendered images to detect visual differences
    """
    
    def __init__(self):
        self.comparison_metrics = []
    
    def compare_images(self, image1_path: str, image2_path: str, 
                      threshold: float = 0.01) -> Dict[str, Any]:
        """
        Compare two images and return detailed comparison metrics
        """
        try:
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            # Ensure images are the same size
            if img1.size != img2.size:
                # Resize the second image to match the first
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays for numerical comparison
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)
            
            # Calculate absolute difference
            diff = np.abs(arr1 - arr2)
            
            # Calculate various metrics
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff))
            
            # Calculate similarity as 1 - normalized mean difference
            # Normalize by the maximum possible difference (255 for 8-bit images)
            normalized_mean_diff = mean_diff / 255.0
            similarity_score = 1.0 - normalized_mean_diff
            
            # Count pixels with differences above threshold
            threshold_abs = threshold * 255  # Convert relative threshold to absolute
            pixel_diff_count = int(np.sum(diff > threshold_abs))
            
            # Calculate percentage of different pixels
            total_pixels = diff.size
            diff_percentage = pixel_diff_count / total_pixels if total_pixels > 0 else 0
            
            result = {
                'similarity_score': similarity_score,
                'mean_difference': mean_diff,
                'max_difference': max_diff,
                'std_difference': std_diff,
                'pixel_difference_count': pixel_diff_count,
                'total_pixels': total_pixels,
                'difference_percentage': diff_percentage,
                'threshold_used': threshold,
                'within_threshold': diff_percentage <= threshold,
                'image1_size': img1.size,
                'image2_size': img2.size
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'similarity_score': 0.0,
                'within_threshold': False
            }
    
    def generate_difference_image(self, image1_path: str, image2_path: str, 
                                 output_path: str) -> bool:
        """
        Generate an image highlighting the differences between two images
        """
        try:
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            # Resize second image if needed
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
            
            # Calculate difference using PIL
            diff_img = ImageChops.difference(img1, img2)
            
            # Enhance the differences
            diff_img = ImageStat.Stat(diff_img)
            # The diff_img here is just the statistics, we need to recalculate
            # as ImageChops.difference returns the actual difference image
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
            
            # Now create the actual difference image
            diff_image = ImageChops.difference(img1, img2)
            diff_image.save(output_path)
            
            return True
        except Exception:
            return False
    
    def detect_visual_artifacts(self, image_path: str, baseline_path: Optional[str] = None) -> List[str]:
        """
        Detect visual artifacts in an image
        """
        artifacts = []
        
        try:
            img = Image.open(image_path).convert('RGB')
            arr = np.array(img)
            
            # Check for single color bands (could indicate rendering issues)
            if len(arr.shape) == 3:  # Color image (height, width, channels)
                for channel in range(arr.shape[2]):
                    channel_data = arr[:, :, channel]
                    # Check if any row or column is completely uniform
                    for i in range(channel_data.shape[0]):
                        if np.all(channel_data[i, :] == channel_data[i, 0]):
                            artifacts.append(f"Uniform row {i} in channel {channel}")
                    for j in range(channel_data.shape[1]):
                        if np.all(channel_data[:, j] == channel_data[0, j]):
                            artifacts.append(f"Uniform column {j} in channel {channel}")
            
            # Check for extreme pixel values that might indicate artifacts
            if np.any(arr > 250) or np.any(arr < 5):
                artifacts.append("Extremely bright or dark pixels detected")
            
            # Check for noise patterns
            # Calculate local variance to detect potential noise
            if arr.shape[0] > 10 and arr.shape[1] > 10:
                # Calculate a simple variance in small regions
                region_size = min(10, arr.shape[0] // 10, arr.shape[1] // 10)
                if region_size > 1:
                    for i in range(0, arr.shape[0] - region_size, region_size):
                        for j in range(0, arr.shape[1] - region_size, region_size):
                            region = arr[i:i+region_size, j:j+region_size]
                            if len(region.shape) == 3:  # Color region
                                variances = [np.var(region[:, :, c]) for c in range(region.shape[2])]
                                if any(v > 5000 for v in variances):  # High variance might indicate noise
                                    artifacts.append(f"High variance noise detected at region ({i},{j})")
            
            # If baseline is provided, compare for artifacts that might have emerged
            if baseline_path:
                baseline_comparison = self.compare_images(baseline_path, image_path, threshold=0.0)
                if baseline_comparison.get('max_difference', 0) > 50:  # Arbitrary threshold
                    artifacts.append("Significant color differences from baseline detected")
        
        except Exception as e:
            artifacts.append(f"Error during artifact detection: {str(e)}")
        
        return artifacts


class ScreenshotManager:
    """
    System for capturing and managing screenshots of shader outputs
    """
    
    def __init__(self, base_dir: str = "screenshots"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.screenshot_counter = 0
    
    def capture_screenshot(self, shader_output: Any, name: str = None) -> str:
        """
        Capture a screenshot of a shader output (simulated)
        """
        # In a real system, this would capture the actual rendered output
        # For simulation, we'll create a placeholder image
        
        if name is None:
            name = f"screenshot_{self.screenshot_counter:04d}"
            self.screenshot_counter += 1
        
        if isinstance(shader_output, np.ndarray):
            # If it's already an array of pixel data, convert it
            img = Image.fromarray(np.uint8(shader_output), 'RGB')
        else:
            # Create a simulated screenshot
            width, height = 800, 600  # Default size
            # Create a gradient image for simulation
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    gradient[i, j] = [(j * 255) // width, (i * 255) // height, 128]
            
            img = Image.fromarray(gradient, 'RGB')
        
        # Save the image
        file_path = self.base_dir / f"{name}.png"
        img.save(file_path)
        
        return str(file_path)
    
    def save_render_output(self, pixel_data: np.ndarray, name: str) -> str:
        """
        Save raw pixel data as an image
        """
        # Ensure pixel data is in the right format
        if pixel_data.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            pixel_data = np.clip(pixel_data, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(pixel_data, 'RGB')
        file_path = self.base_dir / f"{name}.png"
        img.save(file_path)
        
        return str(file_path)
    
    def get_screenshot_path(self, name: str) -> str:
        """
        Get the full path for a screenshot by name
        """
        return str(self.base_dir / f"{name}.png")
    
    def list_screenshots(self) -> List[str]:
        """
        List all screenshot files
        """
        return [str(p) for p in self.base_dir.glob("*.png")]


class VisualRegressionTester:
    """
    Main system for visual regression testing
    """
    
    def __init__(self, storage_dir: str = "visual_tests"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.image_comparator = ImageComparator()
        self.screenshot_manager = ScreenshotManager(str(self.storage_dir / "screenshots"))
        self.test_results: List[VisualTestResult] = []
        self.reference_images: Dict[str, str] = {}  # shader_name -> image_path
    
    def establish_baseline(self, shader_name: str, image_path: str) -> bool:
        """
        Establish a baseline image for a shader
        """
        try:
            # Verify the image exists and is accessible
            if not os.path.exists(image_path):
                return False
            
            # Copy the image to the reference location
            reference_path = str(self.storage_dir / f"baseline_{shader_name}.png")
            Image.open(image_path).save(reference_path)
            
            self.reference_images[shader_name] = reference_path
            return True
        except Exception:
            return False
    
    def set_reference_image(self, shader_name: str, pixel_data: np.ndarray) -> bool:
        """
        Set a reference image for a shader from pixel data
        """
        try:
            # Save the pixel data as the reference image
            reference_path = str(self.storage_dir / f"baseline_{shader_name}.png")
            
            # Ensure pixel data is in the right format
            if pixel_data.dtype != np.uint8:
                pixel_data = np.clip(pixel_data, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(pixel_data, 'RGB')
            img.save(reference_path)
            
            self.reference_images[shader_name] = reference_path
            return True
        except Exception:
            return False
    
    def run_visual_test(self, shader_name: str, current_image_path: str, 
                       threshold: float = 0.01) -> VisualTestResult:
        """
        Run a visual regression test comparing current output to baseline
        """
        import time
        start_time = time.time()
        
        if shader_name not in self.reference_images:
            # No baseline exists, establish current as baseline and pass the test
            self.establish_baseline(shader_name, current_image_path)
            execution_time = time.time() - start_time
            
            result = VisualTestResult(
                test_id=f"test_{len(self.test_results)}",
                shader_name=shader_name,
                baseline_image_path=current_image_path,
                current_image_path=current_image_path,
                passed=True,
                similarity_score=1.0,
                pixel_difference_count=0,
                max_difference=0.0,
                mean_difference=0.0,
                execution_time=execution_time,
                artifacts_detected=[]
            )
            
            self.test_results.append(result)
            return result
        
        # Compare with baseline
        baseline_path = self.reference_images[shader_name]
        comparison = self.image_comparator.compare_images(baseline_path, current_image_path, threshold)
        
        # Detect artifacts in the current image
        artifacts = self.image_comparator.detect_visual_artifacts(current_image_path, baseline_path)
        
        execution_time = time.time() - start_time
        
        result = VisualTestResult(
            test_id=f"test_{len(self.test_results)}",
            shader_name=shader_name,
            baseline_image_path=baseline_path,
            current_image_path=current_image_path,
            passed=comparison.get('within_threshold', False),
            similarity_score=comparison.get('similarity_score', 0.0),
            pixel_difference_count=comparison.get('pixel_difference_count', 0),
            max_difference=comparison.get('max_difference', 0.0),
            mean_difference=comparison.get('mean_difference', 0.0),
            execution_time=execution_time,
            artifacts_detected=artifacts
        )
        
        self.test_results.append(result)
        return result
    
    def run_visual_test_with_capture(self, shader_name: str, shader_output: Any, 
                                   threshold: float = 0.01) -> VisualTestResult:
        """
        Run a visual test by capturing the shader output and comparing to baseline
        """
        # Capture the current output as an image
        current_image_path = self.screenshot_manager.capture_screenshot(shader_output, 
                                                                       f"current_{shader_name}")
        
        # Run the visual test
        return self.run_visual_test(shader_name, current_image_path, threshold)
    
    def generate_difference_visualization(self, shader_name: str, output_dir: str = None) -> Optional[str]:
        """
        Generate a visualization showing differences between baseline and current output
        """
        if shader_name not in self.reference_images:
            return None
        
        if output_dir is None:
            output_dir = self.storage_dir
        
        baseline_path = self.reference_images[shader_name]
        
        # Find the most recent test result for this shader
        recent_result = None
        for result in reversed(self.test_results):
            if result.shader_name == shader_name:
                recent_result = result
                break
        
        if recent_result is None:
            return None
        
        # Create difference visualization
        diff_output_path = str(Path(output_dir) / f"diff_{shader_name}.png")
        success = self.image_comparator.generate_difference_image(
            baseline_path, recent_result.current_image_path, diff_output_path
        )
        
        return diff_output_path if success else None
    
    def get_visual_test_report(self) -> Dict[str, Any]:
        """
        Generate a report on visual regression testing
        """
        if not self.test_results:
            return {"message": "No visual test results available"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate averages
        avg_similarity = sum(r.similarity_score for r in self.test_results) / total_tests
        avg_exec_time = sum(r.execution_time for r in self.test_results) / total_tests
        
        # Find tests with artifacts
        tests_with_artifacts = [r for r in self.test_results if r.artifacts_detected]
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_similarity': avg_similarity,
            'average_execution_time': avg_exec_time,
            'tests_with_artifacts': len(tests_with_artifacts),
            'artifact_examples': [
                {
                    'test_id': r.test_id,
                    'shader_name': r.shader_name,
                    'artifacts': r.artifacts_detected
                }
                for r in tests_with_artifacts[:5]  # First 5 examples
            ],
            'recent_results': [
                {
                    'test_id': r.test_id,
                    'shader_name': r.shader_name,
                    'passed': r.passed,
                    'similarity': r.similarity_score
                }
                for r in self.test_results[-10:]  # Last 10 results
            ]
        }
    
    def capture_and_test_shader(self, shader_name: str, shader_output: Any, 
                               threshold: float = 0.01) -> Dict[str, Any]:
        """
        Convenience method to capture shader output and run visual test
        """
        # Run the visual test
        result = self.run_visual_test_with_capture(shader_name, shader_output, threshold)
        
        # Generate difference visualization if test failed
        diff_path = None
        if not result.passed:
            diff_path = self.generate_difference_visualization(shader_name)
        
        return {
            'test_result': result,
            'difference_image': diff_path,
            'artifacts_detected': result.artifacts_detected
        }


def simulate_shader_output(shader_params: Dict[str, Any]) -> np.ndarray:
    """
    Simulate a shader output as pixel data
    """
    width = shader_params.get('width', 400)
    height = shader_params.get('height', 300)
    
    # Create a simulated shader output
    output = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a pattern based on parameters
    time_param = shader_params.get('time', 0.0)
    scale = shader_params.get('scale', 1.0)
    
    for y in range(height):
        for x in range(width):
            # Simulate some procedural pattern
            value = (np.sin((x * scale + time_param) * 0.02) + 
                    np.cos((y * scale + time_param) * 0.02)) * 127.5 + 127.5
            output[y, x] = [int(value % 256), int((value * 0.7) % 256), int((value * 0.3) % 256)]
    
    return output


def main():
    """
    Example usage of the Visual Regression Testing System
    """
    print("Visual Regression Testing System")
    print("Part of SuperShader Project - Phase 8")
    
    # Create the visual regression testing system
    vis_tester = VisualRegressionTester()
    
    # Simulate shader outputs
    shader_params1 = {'time': 0.0, 'width': 400, 'height': 300, 'scale': 1.0}
    shader_params2 = {'time': 1.0, 'width': 400, 'height': 300, 'scale': 1.0}
    shader_params3 = {'time': 2.0, 'width': 400, 'height': 300, 'scale': 1.0}
    
    # Create initial reference (baseline) output
    baseline_output = simulate_shader_output(shader_params1)
    vis_tester.set_reference_image("test_shader", baseline_output)
    print("Established baseline image for test_shader")
    
    # Test with the same parameters (should pass)
    result1 = vis_tester.capture_and_test_shader("test_shader", baseline_output)
    print(f"Test 1 (same as baseline) - Passed: {result1['test_result'].passed}, "
          f"Similarity: {result1['test_result'].similarity_score:.3f}")
    
    # Test with slightly different parameters
    diff_output = simulate_shader_output(shader_params2)
    result2 = vis_tester.capture_and_test_shader("test_shader", diff_output)
    print(f"Test 2 (time=1.0) - Passed: {result2['test_result'].passed}, "
          f"Similarity: {result2['test_result'].similarity_score:.3f}")
    
    # Test with more different parameters
    diff_output2 = simulate_shader_output(shader_params3)
    result3 = vis_tester.capture_and_test_shader("test_shader", diff_output2)
    print(f"Test 3 (time=2.0) - Passed: {result3['test_result'].passed}, "
          f"Similarity: {result3['test_result'].similarity_score:.3f}")
    
    # Test an entirely different shader
    other_shader_params = {'time': 0.0, 'width': 400, 'height': 300, 'scale': 2.0}
    other_output = simulate_shader_output(other_shader_params)
    result4 = vis_tester.capture_and_test_shader("other_shader", other_output)
    print(f"Test 4 (other_shader) - Passed: {result4['test_result'].passed}, "
          f"Similarity: {result4['test_result'].similarity_score:.3f}")
    
    # Generate and print the test report
    print("\n--- Visual Test Report ---")
    report = vis_tester.get_visual_test_report()
    print(f"Total tests: {report['total_tests']}")
    print(f"Pass rate: {report['success_rate']:.2%}")
    print(f"Average similarity: {report['average_similarity']:.3f}")
    print(f"Tests with artifacts: {report['tests_with_artifacts']}")
    
    if report['artifact_examples']:
        print("Example artifacts detected:")
        for example in report['artifact_examples']:
            print(f"  - {example['shader_name']}: {len(example['artifacts'])} artifacts")


if __name__ == "__main__":
    main()