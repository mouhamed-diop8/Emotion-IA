#!/usr/bin/env python3
"""
Demo Script for DQN Video Parameter Regulation System
Quick demonstration and validation of the complete system
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dqn_video_regulator import VideoParameterEnvironment, DQNAgent
import random

def demo_parameter_optimization(classifier_path, image_path, save_path='demo_optimization.png'):
    """
    Demonstrate parameter optimization on a single image
    
    Args:
        classifier_path: Path to trained classifier
        image_path: Path to test image
        save_path: Path to save visualization
    """
    print("=" * 60)
    print("DEMO: PARAMETER OPTIMIZATION ON SINGLE IMAGE")
    print("=" * 60)
    
    # Load image
    image = Image.open(image_path)
    print(f"\nLoaded image: {image_path}")
    
    # Create environment
    env = VideoParameterEnvironment(classifier_path)
    
    # Reset with image
    state = env.reset(image)
    initial_confidence = env.current_confidence
    
    print(f"Initial confidence: {initial_confidence:.4f}")
    print(f"Initial parameters: contrast={env.contrast:.2f}, exposure={env.exposure:.2f}")
    
    # Try different parameter combinations
    contrast_range = np.linspace(0.5, 2.0, 10)
    exposure_range = np.linspace(0.5, 2.0, 10)
    
    confidence_grid = np.zeros((len(contrast_range), len(exposure_range)))
    
    print("\nExploring parameter space...")
    for i, contrast in enumerate(contrast_range):
        for j, exposure in enumerate(exposure_range):
            env.contrast = contrast
            env.exposure = exposure
            confidence = env._get_classification_confidence()
            confidence_grid[i, j] = confidence
    
    # Find optimal parameters
    max_idx = np.unravel_index(np.argmax(confidence_grid), confidence_grid.shape)
    optimal_contrast = contrast_range[max_idx[0]]
    optimal_exposure = exposure_range[max_idx[1]]
    optimal_confidence = confidence_grid[max_idx]
    
    print(f"\nOptimal parameters found:")
    print(f"  Contrast: {optimal_contrast:.2f}")
    print(f"  Exposure: {optimal_exposure:.2f}")
    print(f"  Confidence: {optimal_confidence:.4f}")
    print(f"  Improvement: {optimal_confidence - initial_confidence:.4f} ({(optimal_confidence - initial_confidence)*100:.2f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original Image\nConfidence: {initial_confidence:.3f}')
    axes[0].axis('off')
    
    # Optimized image
    env.contrast = optimal_contrast
    env.exposure = optimal_exposure
    optimized_img = env._adjust_image(image, optimal_contrast, optimal_exposure)
    axes[1].imshow(optimized_img, cmap='gray')
    axes[1].set_title(f'Optimized Image\nConfidence: {optimal_confidence:.3f}\nC={optimal_contrast:.2f}, E={optimal_exposure:.2f}')
    axes[1].axis('off')
    
    # Confidence heatmap
    im = axes[2].imshow(confidence_grid.T, origin='lower', aspect='auto', cmap='RdYlGn',
                        extent=[contrast_range[0], contrast_range[-1], 
                               exposure_range[0], exposure_range[-1]])
    axes[2].plot(optimal_contrast, optimal_exposure, 'r*', markersize=20, label='Optimal')
    axes[2].plot(1.0, 1.0, 'b*', markersize=15, label='Default')
    axes[2].set_xlabel('Contrast')
    axes[2].set_ylabel('Exposure')
    axes[2].set_title('Confidence Heatmap')
    axes[2].legend()
    plt.colorbar(im, ax=axes[2], label='Confidence')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    return {
        'initial_confidence': initial_confidence,
        'optimal_confidence': optimal_confidence,
        'improvement': optimal_confidence - initial_confidence,
        'optimal_contrast': optimal_contrast,
        'optimal_exposure': optimal_exposure
    }


def demo_dqn_agent(classifier_path, dqn_path, image_path, save_path='demo_dqn_agent.png'):
    """
    Demonstrate DQN agent optimization process
    
    Args:
        classifier_path: Path to trained classifier
        dqn_path: Path to trained DQN agent
        image_path: Path to test image
        save_path: Path to save visualization
    """
    print("\n" + "=" * 60)
    print("DEMO: DQN AGENT OPTIMIZATION PROCESS")
    print("=" * 60)
    
    # Load image
    image = Image.open(image_path)
    print(f"\nLoaded image: {image_path}")
    
    # Create environment and agent
    env = VideoParameterEnvironment(classifier_path)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(dqn_path)
    
    # Reset environment
    state = env.reset(image)
    
    # Track optimization process
    history = {
        'step': [0],
        'contrast': [env.contrast],
        'exposure': [env.exposure],
        'confidence': [env.current_confidence],
        'action': []
    }
    
    print(f"\nInitial state:")
    print(f"  Confidence: {env.current_confidence:.4f}")
    print(f"  Contrast: {env.contrast:.2f}")
    print(f"  Exposure: {env.exposure:.2f}")
    
    print("\nOptimization steps:")
    
    # Run optimization
    max_steps = 15
    for step in range(max_steps):
        # Get action from agent
        action = agent.act(state, training=False)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Record history
        history['step'].append(step + 1)
        history['contrast'].append(info['contrast'])
        history['exposure'].append(info['exposure'])
        history['confidence'].append(info['confidence'])
        history['action'].append(action)
        
        print(f"  Step {step+1}: Action={action}, Confidence={info['confidence']:.4f}, "
              f"Contrast={info['contrast']:.2f}, Exposure={info['exposure']:.2f}, Reward={reward:.3f}")
        
        state = next_state
        
        if done:
            print(f"\n✓ Optimization completed at step {step+1}")
            break
    
    # Final results
    print(f"\nFinal state:")
    print(f"  Confidence: {history['confidence'][-1]:.4f}")
    print(f"  Contrast: {history['contrast'][-1]:.2f}")
    print(f"  Exposure: {history['exposure'][-1]:.2f}")
    print(f"  Improvement: {history['confidence'][-1] - history['confidence'][0]:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confidence over steps
    axes[0, 0].plot(history['step'], history['confidence'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(history['confidence'][0], color='r', linestyle='--', label='Initial')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_title('Confidence Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameters over steps
    axes[0, 1].plot(history['step'], history['contrast'], 'o-', label='Contrast', linewidth=2)
    axes[0, 1].plot(history['step'], history['exposure'], 's-', label='Exposure', linewidth=2)
    axes[0, 1].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].set_title('Parameter Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Parameter trajectory
    axes[1, 0].plot(history['contrast'], history['exposure'], 'o-', linewidth=2, markersize=8)
    axes[1, 0].plot(history['contrast'][0], history['exposure'][0], 'go', markersize=15, label='Start')
    axes[1, 0].plot(history['contrast'][-1], history['exposure'][-1], 'r*', markersize=20, label='End')
    axes[1, 0].set_xlabel('Contrast')
    axes[1, 0].set_ylabel('Exposure')
    axes[1, 0].set_title('Parameter Trajectory')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Action distribution
    if history['action']:
        action_counts = np.bincount(history['action'], minlength=9)
        axes[1, 1].bar(range(9), action_counts, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Action')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].set_xticks(range(9))
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    return history


def main():
    """
    Main demo function
    """
    print("=" * 60)
    print("DQN VIDEO PARAMETER REGULATION - DEMO")
    print("=" * 60)
    
    # Configuration
    CLASSIFIER_MODEL = 'best_model.h5'
    DQN_MODEL = 'dqn_agent.h5'
    
    # Check if models exist
    if not os.path.exists(CLASSIFIER_MODEL):
        print(f"\n❌ Classifier model not found: {CLASSIFIER_MODEL}")
        print("Please train the classifier first using: python3 run_facial_recognition_simple.py")
        return
    
    print(f"\n✓ Found classifier model: {CLASSIFIER_MODEL}")
    
    # Find a test image
    test_image = None
    for root, dirs, files in os.walk('test'):
        for file in files:
            if file.endswith('.jpg'):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        # Try train directory
        for root, dirs, files in os.walk('train'):
            for file in files:
                if file.endswith('.jpg'):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("\n❌ No test images found!")
        return
    
    print(f"✓ Found test image: {test_image}")
    
    # Demo 1: Parameter space exploration
    print("\n" + "=" * 60)
    print("DEMO 1: PARAMETER SPACE EXPLORATION")
    print("=" * 60)
    
    results = demo_parameter_optimization(CLASSIFIER_MODEL, test_image)
    
    # Demo 2: DQN agent (if available)
    if os.path.exists(DQN_MODEL):
        print(f"\n✓ Found DQN model: {DQN_MODEL}")
        history = demo_dqn_agent(CLASSIFIER_MODEL, DQN_MODEL, test_image)
    else:
        print(f"\n⚠ DQN model not found: {DQN_MODEL}")
        print("Train the DQN agent first using: python3 train_dqn_system.py")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - demo_optimization.png (parameter space exploration)")
    if os.path.exists(DQN_MODEL):
        print("  - demo_dqn_agent.png (DQN optimization process)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
