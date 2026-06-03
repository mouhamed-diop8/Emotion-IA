#!/usr/bin/env python3
"""
DQN Agent for Dynamic Video Stream Parameter Regulation
Regulates contrast and exposure to maximize facial expression classification confidence
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import cv2
from PIL import Image, ImageEnhance
import os

class VideoParameterEnvironment:
    """
    Environment for adjusting video parameters (contrast and exposure)
    State: current contrast and exposure values + classification confidence
    Action: adjust contrast/exposure (increase/decrease/maintain)
    Reward: change in classification confidence
    """
    
    def __init__(self, classifier_model_path, img_dim=48):
        """
        Initialize the environment
        
        Args:
            classifier_model_path: Path to trained facial expression classifier
            img_dim: Image dimension for the classifier
        """
        self.img_dim = img_dim
        self.classifier = load_model(classifier_model_path)
        
        # Parameter ranges
        self.contrast_min = 0.5
        self.contrast_max = 2.0
        self.exposure_min = 0.5
        self.exposure_max = 2.0
        
        # Current parameters
        self.contrast = 1.0
        self.exposure = 1.0
        
        # Step size for parameter adjustments
        self.step_size = 0.1
        
        # State space: [contrast, exposure, confidence]
        self.state_size = 3
        
        # Action space: 9 actions (3x3 grid for contrast and exposure)
        # 0: decrease both, 1: decrease contrast/maintain exposure, 2: decrease contrast/increase exposure
        # 3: maintain contrast/decrease exposure, 4: maintain both, 5: maintain contrast/increase exposure
        # 6: increase contrast/decrease exposure, 7: increase contrast/maintain exposure, 8: increase both
        self.action_size = 9
        
        self.current_image = None
        self.current_confidence = 0.0
        
    def reset(self, image):
        """
        Reset environment with a new image
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Initial state
        """
        if isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image)
        else:
            self.current_image = image
            
        # Reset parameters to default
        self.contrast = 1.0
        self.exposure = 1.0
        
        # Get initial confidence
        self.current_confidence = self._get_classification_confidence()
        
        return self._get_state()
    
    def _adjust_image(self, image, contrast, exposure):
        """
        Apply contrast and exposure adjustments to image
        
        Args:
            image: PIL Image
            contrast: Contrast factor
            exposure: Exposure (brightness) factor
            
        Returns:
            Adjusted PIL Image
        """
        # Apply contrast
        enhancer = ImageEnhance.Contrast(image)
        img_adjusted = enhancer.enhance(contrast)
        
        # Apply exposure (brightness)
        enhancer = ImageEnhance.Brightness(img_adjusted)
        img_adjusted = enhancer.enhance(exposure)
        
        return img_adjusted
    
    def _get_classification_confidence(self):
        """
        Get classification confidence for current image with current parameters
        
        Returns:
            Maximum confidence score from classifier
        """
        # Apply current adjustments
        adjusted_img = self._adjust_image(self.current_image, self.contrast, self.exposure)
        
        # Convert to grayscale and resize
        adjusted_img = adjusted_img.convert('L')
        adjusted_img = adjusted_img.resize((self.img_dim, self.img_dim))
        
        # Prepare for classifier
        img_array = np.array(adjusted_img) / 255.0
        img_array = img_array.reshape(1, self.img_dim, self.img_dim, 1)
        
        # Get prediction
        predictions = self.classifier.predict(img_array, verbose=0)
        confidence = np.max(predictions)
        
        return confidence
    
    def _get_state(self):
        """
        Get current state
        
        Returns:
            State array [contrast, exposure, confidence]
        """
        return np.array([self.contrast, self.exposure, self.current_confidence])
    
    def step(self, action):
        """
        Execute action and return new state, reward, done flag
        
        Args:
            action: Action index (0-8)
            
        Returns:
            next_state, reward, done, info
        """
        # Store previous confidence
        prev_confidence = self.current_confidence
        
        # Decode action
        contrast_action = action // 3  # 0: decrease, 1: maintain, 2: increase
        exposure_action = action % 3   # 0: decrease, 1: maintain, 2: increase
        
        # Apply contrast adjustment
        if contrast_action == 0:
            self.contrast = max(self.contrast_min, self.contrast - self.step_size)
        elif contrast_action == 2:
            self.contrast = min(self.contrast_max, self.contrast + self.step_size)
        
        # Apply exposure adjustment
        if exposure_action == 0:
            self.exposure = max(self.exposure_min, self.exposure - self.step_size)
        elif exposure_action == 2:
            self.exposure = min(self.exposure_max, self.exposure + self.step_size)
        
        # Get new confidence
        self.current_confidence = self._get_classification_confidence()
        
        # Calculate reward (improvement in confidence)
        reward = (self.current_confidence - prev_confidence) * 10.0
        
        # Add penalty for extreme parameter values
        if self.contrast < 0.7 or self.contrast > 1.5:
            reward -= 0.1
        if self.exposure < 0.7 or self.exposure > 1.5:
            reward -= 0.1
        
        # Episode is done if confidence is very high or parameters are at extremes
        done = self.current_confidence > 0.95 or \
               (self.contrast <= self.contrast_min and self.exposure <= self.exposure_min) or \
               (self.contrast >= self.contrast_max and self.exposure >= self.exposure_max)
        
        next_state = self._get_state()
        info = {
            'contrast': self.contrast,
            'exposure': self.exposure,
            'confidence': self.current_confidence,
            'confidence_improvement': self.current_confidence - prev_confidence
        }
        
        return next_state, reward, done, info


class DQNAgent:
    """
    Deep Q-Network Agent for learning optimal parameter adjustments
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # Experience replay buffer
        self.memory = deque(maxlen=2000)
        
        # Q-Network
        self.model = self._build_model()
        
        # Target Network (for stable learning)
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """
        Build neural network for Q-value approximation
        
        Returns:
            Keras model
        """
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """
        Copy weights from main model to target model
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Action index
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train on a batch of experiences from replay buffer
        
        Returns:
            Average loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        current_q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values with Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def save(self, filepath):
        """
        Save model weights
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model weights
        """
        self.model = load_model(filepath)
        self.update_target_model()
        print(f"Model loaded from {filepath}")


def train_dqn_agent(classifier_model_path, training_images_dir, episodes=100, 
                    max_steps_per_episode=20, save_path='dqn_agent.h5'):
    """
    Train DQN agent on a set of training images
    
    Args:
        classifier_model_path: Path to trained facial expression classifier
        training_images_dir: Directory containing training images
        episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_path: Path to save trained agent
    """
    print("=" * 60)
    print("TRAINING DQN AGENT FOR VIDEO PARAMETER REGULATION")
    print("=" * 60)
    
    # Initialize environment and agent
    env = VideoParameterEnvironment(classifier_model_path)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Get list of training images
    image_files = []
    for root, dirs, files in os.walk(training_images_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(image_files)} training images")
    print(f"Training for {episodes} episodes with max {max_steps_per_episode} steps per episode\n")
    
    # Training metrics
    episode_rewards = []
    episode_confidences = []
    losses = []
    
    for episode in range(episodes):
        # Select random image
        image_path = random.choice(image_files)
        image = Image.open(image_path)
        
        # Reset environment
        state = env.reset(image)
        total_reward = 0
        initial_confidence = env.current_confidence
        
        for step in range(max_steps_per_episode):
            # Choose action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                losses.append(loss)
            
            if done:
                break
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_confidences.append(info['confidence'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_confidence = np.mean(episode_confidences[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.4f}")
            print(f"  Avg Confidence (last 10): {avg_confidence:.4f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Initial Confidence: {initial_confidence:.4f} -> Final: {info['confidence']:.4f}")
            print()
    
    # Save trained agent
    agent.save(save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"\nFinal Statistics:")
    print(f"  Average Reward: {np.mean(episode_rewards):.4f}")
    print(f"  Average Final Confidence: {np.mean(episode_confidences):.4f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    
    return agent, episode_rewards, episode_confidences, losses


if __name__ == "__main__":
    # Example usage
    print("DQN Video Parameter Regulator Module")
    print("This module provides DQN-based dynamic parameter regulation")
    print("Import this module to use the VideoParameterEnvironment and DQNAgent classes")

# Made with Bob
