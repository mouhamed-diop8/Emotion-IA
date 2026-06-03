#!/usr/bin/env python3
"""
Real-time Video Processing with DQN-based Parameter Regulation
Processes video stream with dynamic contrast/exposure adjustments
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import argparse
import os

class RealtimeVideoProcessor:
    """
    Real-time video processor with DQN-based parameter regulation
    """
    
    def __init__(self, classifier_path, dqn_agent_path, img_dim=48):
        """
        Initialize the video processor
        
        Args:
            classifier_path: Path to facial expression classifier
            dqn_agent_path: Path to trained DQN agent
            img_dim: Image dimension for classifier
        """
        print("Loading models...")
        self.classifier = load_model(classifier_path)
        self.dqn_agent = load_model(dqn_agent_path)
        self.img_dim = img_dim
        
        # Emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Current parameters
        self.contrast = 1.0
        self.exposure = 1.0
        
        # Parameter ranges
        self.contrast_min = 0.5
        self.contrast_max = 2.0
        self.exposure_min = 0.5
        self.exposure_max = 2.0
        self.step_size = 0.1
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Statistics
        self.frame_count = 0
        self.total_confidence = 0
        self.adjustment_count = 0
        
        print("✓ Models loaded successfully")
    
    def adjust_image(self, image, contrast, exposure):
        """
        Apply contrast and exposure adjustments
        
        Args:
            image: PIL Image
            contrast: Contrast factor
            exposure: Exposure factor
            
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
    
    def get_classification(self, face_img):
        """
        Get emotion classification and confidence
        
        Args:
            face_img: Face image (PIL Image)
            
        Returns:
            emotion, confidence
        """
        # Apply current adjustments
        adjusted_img = self.adjust_image(face_img, self.contrast, self.exposure)
        
        # Convert to grayscale and resize
        adjusted_img = adjusted_img.convert('L')
        adjusted_img = adjusted_img.resize((self.img_dim, self.img_dim))
        
        # Prepare for classifier
        img_array = np.array(adjusted_img) / 255.0
        img_array = img_array.reshape(1, self.img_dim, self.img_dim, 1)
        
        # Get prediction
        predictions = self.classifier.predict(img_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        return self.emotions[emotion_idx], confidence, predictions[0]
    
    def get_dqn_action(self, confidence):
        """
        Get action from DQN agent
        
        Args:
            confidence: Current classification confidence
            
        Returns:
            Action index
        """
        state = np.array([self.contrast, self.exposure, confidence])
        q_values = self.dqn_agent.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def apply_action(self, action):
        """
        Apply DQN action to adjust parameters
        
        Args:
            action: Action index (0-8)
        """
        # Decode action
        contrast_action = action // 3  # 0: decrease, 1: maintain, 2: increase
        exposure_action = action % 3   # 0: decrease, 1: maintain, 2: increase
        
        # Apply contrast adjustment
        if contrast_action == 0:
            self.contrast = max(self.contrast_min, self.contrast - self.step_size)
            self.adjustment_count += 1
        elif contrast_action == 2:
            self.contrast = min(self.contrast_max, self.contrast + self.step_size)
            self.adjustment_count += 1
        
        # Apply exposure adjustment
        if exposure_action == 0:
            self.exposure = max(self.exposure_min, self.exposure - self.step_size)
            self.adjustment_count += 1
        elif exposure_action == 2:
            self.exposure = min(self.exposure_max, self.exposure + self.step_size)
            self.adjustment_count += 1
    
    def process_frame(self, frame, use_dqn=True):
        """
        Process a single frame
        
        Args:
            frame: Video frame (numpy array)
            use_dqn: Whether to use DQN for parameter adjustment
            
        Returns:
            Processed frame with annotations
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
            # Get classification
            emotion, confidence, all_predictions = self.get_classification(face_pil)
            
            # Update statistics
            self.frame_count += 1
            self.total_confidence += confidence
            
            # Use DQN to adjust parameters
            if use_dqn and self.frame_count % 5 == 0:  # Adjust every 5 frames
                action = self.get_dqn_action(confidence)
                self.apply_action(action)
            
            # Draw rectangle around face
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display emotion and confidence
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Display top 3 predictions
            top_3_idx = np.argsort(all_predictions)[-3:][::-1]
            y_offset = y + h + 20
            for idx in top_3_idx:
                pred_text = f"{self.emotions[idx]}: {all_predictions[idx]:.2f}"
                cv2.putText(frame, pred_text, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
        
        # Display parameters and statistics
        param_text = f"Contrast: {self.contrast:.2f} | Exposure: {self.exposure:.2f}"
        cv2.putText(frame, param_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)
        
        if self.frame_count > 0:
            avg_conf = self.total_confidence / self.frame_count
            stats_text = f"Avg Confidence: {avg_conf:.3f} | Adjustments: {self.adjustment_count}"
            cv2.putText(frame, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
        
        # Display mode
        mode_text = "DQN: ON" if use_dqn else "DQN: OFF"
        cv2.putText(frame, mode_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_source=0, use_dqn=True, output_path=None):
        """
        Process video stream
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            use_dqn: Whether to use DQN for parameter adjustment
            output_path: Path to save output video (optional)
        """
        print(f"\nOpening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording to: {output_path}")
        
        print("\nProcessing video...")
        print("Press 'q' to quit")
        print("Press 'd' to toggle DQN on/off")
        print("Press 'r' to reset parameters")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("End of video or error reading frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame, use_dqn)
                
                # Write to output if specified
                if writer:
                    writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('DQN Video Parameter Regulation', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('d'):
                    use_dqn = not use_dqn
                    print(f"\nDQN {'enabled' if use_dqn else 'disabled'}")
                elif key == ord('r'):
                    self.contrast = 1.0
                    self.exposure = 1.0
                    print("\nParameters reset to default")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            elapsed_time = time.time() - start_time
            
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "=" * 60)
            print("PROCESSING STATISTICS")
            print("=" * 60)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Processing time: {elapsed_time:.2f} seconds")
            if self.frame_count > 0:
                print(f"Average FPS: {self.frame_count / elapsed_time:.2f}")
                print(f"Average confidence: {self.total_confidence / self.frame_count:.4f}")
            print(f"Total parameter adjustments: {self.adjustment_count}")
            print(f"Final contrast: {self.contrast:.2f}")
            print(f"Final exposure: {self.exposure:.2f}")
            print("=" * 60)


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description='Real-time video processing with DQN parameter regulation'
    )
    parser.add_argument(
        '--classifier', 
        type=str, 
        default='best_model.h5',
        help='Path to facial expression classifier model'
    )
    parser.add_argument(
        '--dqn', 
        type=str, 
        default='dqn_agent.h5',
        help='Path to trained DQN agent model'
    )
    parser.add_argument(
        '--video', 
        type=str, 
        default='0',
        help='Video source (0 for webcam, or path to video file)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Path to save output video (optional)'
    )
    parser.add_argument(
        '--no-dqn', 
        action='store_true',
        help='Disable DQN (use default parameters)'
    )
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.classifier):
        print(f"❌ Error: Classifier model not found: {args.classifier}")
        print("Please train the classifier first using run_facial_recognition_simple.py")
        return
    
    if not os.path.exists(args.dqn):
        print(f"❌ Error: DQN agent not found: {args.dqn}")
        print("Please train the DQN agent first using train_dqn_system.py")
        return
    
    # Parse video source
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Create processor
    processor = RealtimeVideoProcessor(args.classifier, args.dqn)
    
    # Process video
    processor.process_video(
        video_source=video_source,
        use_dqn=not args.no_dqn,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

# Made with Bob
