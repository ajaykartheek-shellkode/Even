#!/usr/bin/env python3
"""
Test script to simulate optimization progress for UI testing.
"""

import time
from progress_tracker import start_tracking, update_trial_progress, complete_tracking, clear_tracking

def simulate_optimization():
    """Simulate a 5-trial optimization with realistic progress."""
    
    print("🧪 Starting progress simulation for UI testing...")
    
    # Start tracking
    start_tracking(5, "Starting test optimization")
    
    # Simulate trials
    scores = [0.72, 0.81, 0.79, 0.87, 0.89]
    
    for trial in range(1, 6):
        print(f"⚡ Trial {trial}/5 - Score: {scores[trial-1]:.3f}")
        update_trial_progress(trial, scores[trial-1], f"Running trial {trial}")
        
        # Simulate processing time
        time.sleep(8)  # 8 seconds per trial
    
    # Complete optimization
    complete_tracking(0.89)
    print("✅ Simulation completed! Check the UI for progress updates.")
    
    # Keep progress file for 30 seconds to see results
    print("📊 Progress file will remain for 30 seconds...")
    time.sleep(30)
    
    # Clean up
    clear_tracking()
    print("🧹 Progress tracking cleared.")

if __name__ == "__main__":
    simulate_optimization()