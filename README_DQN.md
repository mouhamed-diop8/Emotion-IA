# DQN-Based Video Parameter Regulation for Facial Expression Recognition

## 📋 Project Overview

This project implements a **Deep Q-Network (DQN)** reinforcement learning agent that dynamically regulates video stream parameters (contrast and exposure) to maximize the confidence score of facial expression classification.

### Key Components

1. **Facial Expression Classifier**: CNN model trained to recognize 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
2. **DQN Agent**: Reinforcement learning agent that learns optimal parameter adjustments
3. **Video Parameter Environment**: Custom RL environment for contrast/exposure regulation
4. **Real-time Video Processor**: Live video processing with dynamic parameter optimization

---

## 🏗️ Architecture

### 1. DQN Agent Architecture

```
State Space (3 dimensions):
- Current contrast value [0.5 - 2.0]
- Current exposure value [0.5 - 2.0]
- Classification confidence [0.0 - 1.0]

Action Space (9 discrete actions):
- 0: Decrease both contrast and exposure
- 1: Decrease contrast, maintain exposure
- 2: Decrease contrast, increase exposure
- 3: Maintain contrast, decrease exposure
- 4: Maintain both (no change)
- 5: Maintain contrast, increase exposure
- 6: Increase contrast, decrease exposure
- 7: Increase contrast, maintain exposure
- 8: Increase both contrast and exposure

Neural Network:
- Input Layer: 3 neurons (state)
- Hidden Layer 1: 64 neurons + Dropout(0.2) + ReLU
- Hidden Layer 2: 64 neurons + Dropout(0.2) + ReLU
- Hidden Layer 3: 32 neurons + ReLU
- Output Layer: 9 neurons (Q-values for each action)
```

### 2. Reward Function

```python
reward = (new_confidence - old_confidence) * 10.0

# Penalties for extreme values
if contrast < 0.7 or contrast > 1.5:
    reward -= 0.1
if exposure < 0.7 or exposure > 1.5:
    reward -= 0.1
```

### 3. Training Process

- **Experience Replay**: Buffer size of 2000 experiences
- **Target Network**: Updated every 10 episodes for stable learning
- **Epsilon-Greedy**: ε starts at 1.0, decays to 0.01 (exploration vs exploitation)
- **Discount Factor (γ)**: 0.95
- **Batch Size**: 32
- **Learning Rate**: 0.001

---

## 📁 Project Files

### Core Implementation Files

1. **`dqn_video_regulator.py`**
   - `VideoParameterEnvironment`: RL environment for parameter adjustment
   - `DQNAgent`: Deep Q-Network implementation
   - `train_dqn_agent()`: Training function

2. **`train_dqn_system.py`**
   - Complete training pipeline
   - Training and testing visualization
   - Performance metrics generation

3. **`realtime_video_dqn.py`**
   - Real-time video processing
   - Live parameter adjustment using trained DQN
   - Interactive controls

4. **`run_facial_recognition_simple.py`**
   - Facial expression classifier training
   - CNN model architecture
   - Data preprocessing and augmentation

---

## 🚀 Usage Guide

### Step 1: Train Facial Expression Classifier

First, train the base CNN classifier for facial expression recognition:

```bash
python3 run_facial_recognition_simple.py
```

**Output:**
- `best_model.h5`: Best model during training
- `my_cnn_model.h5`: Final trained model
- `training_history.png`: Training metrics visualization

**Expected Training Time:** 30-60 minutes (depending on hardware)

---

### Step 2: Train DQN Agent

Train the DQN agent to learn optimal parameter adjustments:

```bash
python3 train_dqn_system.py
```

**Output:**
- `dqn_agent.h5`: Trained DQN agent
- `dqn_training_results.png`: Training metrics (rewards, confidence, loss)
- `dqn_test_results.png`: Test performance visualization

**Training Configuration:**
- Episodes: 200
- Max steps per episode: 20
- Training images: Randomly sampled from train directory

**Expected Training Time:** 20-40 minutes

---

### Step 3: Real-time Video Processing

Process video with dynamic parameter regulation:

#### Using Webcam:
```bash
python3 realtime_video_dqn.py --video 0
```

#### Using Video File:
```bash
python3 realtime_video_dqn.py --video path/to/video.mp4
```

#### Save Output Video:
```bash
python3 realtime_video_dqn.py --video 0 --output output.mp4
```

#### Disable DQN (baseline comparison):
```bash
python3 realtime_video_dqn.py --video 0 --no-dqn
```

**Interactive Controls:**
- Press `q`: Quit
- Press `d`: Toggle DQN on/off
- Press `r`: Reset parameters to default

---

## 📊 Performance Metrics

### Training Metrics

The system tracks and visualizes:

1. **Episode Rewards**: Total reward accumulated per episode
2. **Classification Confidence**: Final confidence after parameter optimization
3. **Training Loss**: DQN network training loss
4. **Reward Distribution**: Histogram of episode rewards

### Test Metrics

1. **Confidence Improvement**: Initial vs final confidence comparison
2. **Parameter Adjustments**: Visualization of optimal contrast/exposure values
3. **Success Rate**: Percentage of images with improved confidence
4. **Average Steps**: Number of steps to reach optimal parameters

### Real-time Metrics

Displayed during video processing:
- Current contrast and exposure values
- Average classification confidence
- Total parameter adjustments
- Processing FPS

---

## 🔬 Technical Details

### DQN Algorithm

The implementation uses the classic DQN algorithm with:

1. **Experience Replay**: Breaks correlation between consecutive samples
2. **Target Network**: Stabilizes training by using separate network for target Q-values
3. **Epsilon-Greedy Policy**: Balances exploration and exploitation
4. **Bellman Equation**: Updates Q-values based on immediate reward + discounted future reward

### Environment Design

**State Representation:**
- Normalized parameter values and confidence score
- Provides agent with current system state

**Action Space:**
- Discrete actions for interpretability
- 3x3 grid covers all parameter adjustment combinations

**Reward Shaping:**
- Primary reward: Improvement in classification confidence
- Penalty terms: Discourage extreme parameter values
- Encourages finding balanced, optimal parameters

### Image Processing Pipeline

1. **Face Detection**: Haar Cascade classifier
2. **Parameter Adjustment**: PIL ImageEnhance (Contrast, Brightness)
3. **Preprocessing**: Grayscale conversion, resize to 48x48
4. **Classification**: CNN forward pass
5. **DQN Decision**: Q-value computation and action selection

---

## 📈 Expected Results

### Baseline (No DQN)
- Fixed parameters: contrast=1.0, exposure=1.0
- Average confidence: ~0.60-0.70

### With DQN
- Dynamic parameters: optimized per frame
- Average confidence: ~0.75-0.85 (10-15% improvement)
- Adaptive to lighting conditions and image quality

### Key Improvements
- Better performance in poor lighting conditions
- Improved confidence on low-quality images
- Automatic adaptation to different video sources

---

## 🛠️ Requirements

```bash
pip3 install opencv-python tensorflow pandas matplotlib seaborn scikit-learn pillow
```

**Minimum Requirements:**
- Python 3.7+
- TensorFlow 2.x
- OpenCV 4.x
- 8GB RAM (16GB recommended)
- GPU recommended for training (optional)

---

## 📝 Configuration

### Hyperparameters (in `dqn_video_regulator.py`)

```python
# DQN Agent
gamma = 0.95              # Discount factor
epsilon = 1.0             # Initial exploration rate
epsilon_min = 0.01        # Minimum exploration rate
epsilon_decay = 0.995     # Exploration decay rate
batch_size = 32           # Training batch size
learning_rate = 0.001     # Optimizer learning rate

# Environment
contrast_min = 0.5        # Minimum contrast
contrast_max = 2.0        # Maximum contrast
exposure_min = 0.5        # Minimum exposure
exposure_max = 2.0        # Maximum exposure
step_size = 0.1           # Parameter adjustment step
```

### Training Configuration (in `train_dqn_system.py`)

```python
episodes = 200                    # Number of training episodes
max_steps_per_episode = 20        # Maximum steps per episode
num_test_images = 30              # Number of test images
```

---

## 🎯 Use Cases

1. **Security Systems**: Improve face recognition in varying lighting conditions
2. **Video Conferencing**: Enhance facial expression detection quality
3. **Emotion Analysis**: Optimize video quality for emotion recognition systems
4. **Surveillance**: Adaptive video processing for different environments
5. **Research**: Study RL applications in computer vision preprocessing

---

## 🔍 Troubleshooting

### Issue: Low confidence scores
**Solution:** 
- Ensure classifier is properly trained (>60% test accuracy)
- Check if faces are properly detected
- Verify image quality and lighting

### Issue: DQN not improving performance
**Solution:**
- Train for more episodes (increase from 200 to 500)
- Adjust reward function weights
- Check if training images are diverse enough

### Issue: Slow video processing
**Solution:**
- Reduce frame processing frequency (adjust `frame_count % 5`)
- Use GPU acceleration
- Lower video resolution

### Issue: Parameters oscillating
**Solution:**
- Decrease step_size (e.g., from 0.1 to 0.05)
- Increase epsilon_decay for faster convergence
- Add momentum to parameter updates

---

## 📚 References

1. **DQN Paper**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. **Facial Expression Recognition**: FER2013 dataset methodology
3. **OpenCV Documentation**: Face detection and image processing
4. **TensorFlow/Keras**: Deep learning framework documentation

---

## 🤝 Contributing

This project is part of a Deep Learning and Reinforcement Learning course project. 

**Future Improvements:**
- [ ] Implement Double DQN for better stability
- [ ] Add Dueling DQN architecture
- [ ] Extend to more parameters (saturation, sharpness)
- [ ] Multi-face optimization
- [ ] Real-time performance optimization

---

## 📄 License

Educational project - Free to use and modify for learning purposes.

---

## 👨‍💻 Author

Created as part of the "Projet DL&APR" (Deep Learning & Apprentissage par Renforcement)

**Date:** 2026

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Deep Q-Network implementation from scratch
- ✅ Custom RL environment design
- ✅ Integration of RL with computer vision
- ✅ Real-time video processing
- ✅ Hyperparameter tuning and optimization
- ✅ Performance evaluation and visualization

---

**Happy Learning! 🚀**