#!/usr/bin/env python3
"""
Training Script for DQN-based Video Parameter Regulation System
Trains the DQN agent to optimize contrast and exposure for facial expression recognition
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dqn_video_regulator import VideoParameterEnvironment, DQNAgent, train_dqn_agent
from PIL import Image
import random

def plot_training_results(episode_rewards, episode_confidences, losses, save_path='dqn_training_results.png'):
    """
    Plot training metrics
    
    Args:
        episode_rewards: List of episode rewards
        episode_confidences: List of final confidences per episode
        losses: List of training losses
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) > 10:
        # Moving average
        window = 10
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                       'r-', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Classification Confidence
    axes[0, 1].plot(episode_confidences, alpha=0.6, label='Final Confidence')
    if len(episode_confidences) > 10:
        window = 10
        moving_avg = np.convolve(episode_confidences, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_confidences)), moving_avg, 
                       'r-', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Confidence Score')
    axes[0, 1].set_title('Classification Confidence Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Training Loss
    if losses:
        axes[1, 0].plot(losses, alpha=0.4)
        if len(losses) > 50:
            window = 50
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(losses)), moving_avg, 
                           'r-', linewidth=2, label=f'Moving Avg ({window})')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(episode_rewards), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training results plot saved to {save_path}")


def test_dqn_agent(agent, env, test_images_dir, num_test_images=20):
    """
    Test the trained DQN agent on test images
    
    Args:
        agent: Trained DQN agent
        env: Video parameter environment
        test_images_dir: Directory containing test images
        num_test_images: Number of images to test
        
    Returns:
        Test results dictionary
    """
    print("\n" + "=" * 60)
    print("TESTING DQN AGENT")
    print("=" * 60)
    
    # Get test images
    image_files = []
    for root, dirs, files in os.walk(test_images_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if len(image_files) == 0:
        print("❌ No test images found!")
        return None
    
    # Randomly select test images
    test_images = random.sample(image_files, min(num_test_images, len(image_files)))
    
    results = {
        'initial_confidences': [],
        'final_confidences': [],
        'improvements': [],
        'num_steps': [],
        'final_contrasts': [],
        'final_exposures': []
    }
    
    print(f"\nTesting on {len(test_images)} images...\n")
    
    for idx, image_path in enumerate(test_images):
        image = Image.open(image_path)
        state = env.reset(image)
        
        initial_confidence = env.current_confidence
        steps = 0
        max_steps = 20
        
        # Run episode with greedy policy (no exploration)
        for step in range(max_steps):
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Record results
        results['initial_confidences'].append(initial_confidence)
        results['final_confidences'].append(info['confidence'])
        results['improvements'].append(info['confidence'] - initial_confidence)
        results['num_steps'].append(steps)
        results['final_contrasts'].append(info['contrast'])
        results['final_exposures'].append(info['exposure'])
        
        if (idx + 1) % 5 == 0:
            print(f"Tested {idx + 1}/{len(test_images)} images...")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nAverage Initial Confidence: {np.mean(results['initial_confidences']):.4f}")
    print(f"Average Final Confidence: {np.mean(results['final_confidences']):.4f}")
    print(f"Average Improvement: {np.mean(results['improvements']):.4f}")
    print(f"Average Steps Taken: {np.mean(results['num_steps']):.2f}")
    print(f"Average Final Contrast: {np.mean(results['final_contrasts']):.4f}")
    print(f"Average Final Exposure: {np.mean(results['final_exposures']):.4f}")
    
    # Success rate (improvement > 0)
    success_rate = sum(1 for imp in results['improvements'] if imp > 0) / len(results['improvements'])
    print(f"\nSuccess Rate (improved confidence): {success_rate*100:.2f}%")
    
    return results


def plot_test_results(results, save_path='dqn_test_results.png'):
    """
    Plot test results
    
    Args:
        results: Dictionary of test results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Confidence Comparison
    x = range(len(results['initial_confidences']))
    axes[0, 0].plot(x, results['initial_confidences'], 'o-', label='Initial', alpha=0.7)
    axes[0, 0].plot(x, results['final_confidences'], 's-', label='Final', alpha=0.7)
    axes[0, 0].set_xlabel('Test Image Index')
    axes[0, 0].set_ylabel('Confidence Score')
    axes[0, 0].set_title('Confidence: Initial vs Final')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Plot 2: Confidence Improvement
    axes[0, 1].bar(x, results['improvements'], alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Test Image Index')
    axes[0, 1].set_ylabel('Confidence Improvement')
    axes[0, 1].set_title('Confidence Improvement per Image')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Parameter Adjustments
    axes[1, 0].scatter(results['final_contrasts'], results['final_exposures'], 
                      c=results['improvements'], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Final Contrast')
    axes[1, 0].set_ylabel('Final Exposure')
    axes[1, 0].set_title('Parameter Adjustments (color = improvement)')
    axes[1, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Improvement')
    
    # Plot 4: Steps Distribution
    axes[1, 1].hist(results['num_steps'], bins=range(1, max(results['num_steps'])+2), 
                   alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(results['num_steps']), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(results["num_steps"]):.1f}')
    axes[1, 1].set_xlabel('Number of Steps')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Steps Taken to Optimize')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Test results plot saved to {save_path}")


def main():
    """
    Main training and testing pipeline
    """
    print("=" * 60)
    print("DQN VIDEO PARAMETER REGULATION SYSTEM")
    print("=" * 60)
    
    # Configuration
    CLASSIFIER_MODEL = 'best_model.h5'  # Trained facial expression classifier
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    DQN_MODEL_PATH = 'dqn_agent.h5'
    
    # Check if classifier model exists
    if not os.path.exists(CLASSIFIER_MODEL):
        print(f"\n❌ ERROR: Classifier model not found: {CLASSIFIER_MODEL}")
        print("Please train the facial expression classifier first using run_facial_recognition_simple.py")
        return
    
    # Check if training data exists
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ ERROR: Training directory not found: {TRAIN_DIR}")
        return
    
    print(f"\n✓ Found classifier model: {CLASSIFIER_MODEL}")
    print(f"✓ Found training directory: {TRAIN_DIR}")
    
    # Train DQN agent
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING DQN AGENT")
    print("=" * 60)
    
    agent, episode_rewards, episode_confidences, losses = train_dqn_agent(
        classifier_model_path=CLASSIFIER_MODEL,
        training_images_dir=TRAIN_DIR,
        episodes=200,  # Number of training episodes
        max_steps_per_episode=20,
        save_path=DQN_MODEL_PATH
    )
    
    # Plot training results
    plot_training_results(episode_rewards, episode_confidences, losses)
    
    # Test DQN agent
    if os.path.exists(TEST_DIR):
        print("\n" + "=" * 60)
        print("PHASE 2: TESTING DQN AGENT")
        print("=" * 60)
        
        env = VideoParameterEnvironment(CLASSIFIER_MODEL)
        test_results = test_dqn_agent(agent, env, TEST_DIR, num_test_images=30)
        
        if test_results:
            plot_test_results(test_results)
    else:
        print(f"\n⚠ Warning: Test directory not found: {TEST_DIR}")
        print("Skipping testing phase.")
    
    print("\n" + "=" * 60)
    print("TRAINING AND TESTING COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {DQN_MODEL_PATH} (trained DQN agent)")
    print("  - dqn_training_results.png (training metrics)")
    if os.path.exists(TEST_DIR):
        print("  - dqn_test_results.png (test results)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
