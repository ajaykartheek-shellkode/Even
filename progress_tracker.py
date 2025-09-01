#!/usr/bin/env python3
"""
Progress Tracker for Prompt Optimization.
Provides utilities for tracking and updating optimization progress.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class ProgressTracker:
    """
    Tracks optimization progress and writes updates to a JSON file.
    """
    
    def __init__(self, progress_file: str = "optimization_progress.json"):
        self.progress_file = Path(progress_file)
        self.progress_data = {}
        
    def start_optimization(self, total_trials: int, phase: str = "Initializing"):
        """Start tracking a new optimization run."""
        self.progress_data = {
            'status': 'running',
            'start_time': time.time(),
            'current_trial': 0,
            'total_trials': total_trials,
            'best_score_so_far': 0.0,
            'current_score': 0.0,
            'api_calls_made': 0,
            'current_phase': phase,
            'last_updated': time.time()
        }
        self._save_progress()
        print(f"🚀 Started optimization tracking: {total_trials} trials")
    
    def update_trial(self, trial_num: int, score: float = 0.0, phase: str = "Running trial"):
        """Update progress for a specific trial."""
        if not self.progress_data:
            return
            
        self.progress_data.update({
            'current_trial': trial_num,
            'current_score': score,
            'current_phase': phase,
            'last_updated': time.time()
        })
        
        # Update best score if this is better
        if score > self.progress_data.get('best_score_so_far', 0):
            self.progress_data['best_score_so_far'] = score
            print(f"🎯 New best score: {score:.3f} (Trial {trial_num})")
        
        self._save_progress()
    
    def update_api_calls(self, api_calls: int):
        """Update the number of API calls made."""
        if not self.progress_data:
            return
            
        self.progress_data.update({
            'api_calls_made': api_calls,
            'last_updated': time.time()
        })
        self._save_progress()
    
    def update_phase(self, phase: str):
        """Update the current phase description."""
        if not self.progress_data:
            return
            
        self.progress_data.update({
            'current_phase': phase,
            'last_updated': time.time()
        })
        self._save_progress()
    
    def complete_optimization(self, final_score: float, message: str = "Optimization completed"):
        """Mark optimization as completed."""
        if not self.progress_data:
            return
            
        self.progress_data.update({
            'status': 'completed',
            'current_trial': self.progress_data.get('total_trials', 0),
            'current_score': final_score,
            'best_score_so_far': max(final_score, self.progress_data.get('best_score_so_far', 0)),
            'current_phase': message,
            'last_updated': time.time(),
            'end_time': time.time()
        })
        self._save_progress()
        print(f"✅ Optimization completed with score: {final_score:.3f}")
    
    def fail_optimization(self, error_message: str):
        """Mark optimization as failed."""
        if not self.progress_data:
            return
            
        self.progress_data.update({
            'status': 'failed',
            'current_phase': f"Failed: {error_message}",
            'last_updated': time.time(),
            'end_time': time.time()
        })
        self._save_progress()
        print(f"❌ Optimization failed: {error_message}")
    
    def clear_progress(self):
        """Clear progress file (call when optimization is fully done)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            print("🧹 Progress tracking cleared")
    
    def _save_progress(self):
        """Save current progress to file."""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")
    
    def get_current_progress(self) -> Optional[Dict[str, Any]]:
        """Get current progress data."""
        return self.progress_data.copy() if self.progress_data else None


# Global instance for easy importing
tracker = ProgressTracker("progress_state.json")


def start_tracking(total_trials: int, phase: str = "Initializing"):
    """Convenience function to start tracking."""
    tracker.start_optimization(total_trials, phase)


def update_trial_progress(trial_num: int, score: float = 0.0, phase: str = "Running trial"):
    """Convenience function to update trial progress."""
    tracker.update_trial(trial_num, score, phase)


def update_phase(phase: str):
    """Convenience function to update current phase."""
    tracker.update_phase(phase)


def complete_tracking(final_score: float):
    """Convenience function to complete tracking."""
    tracker.complete_optimization(final_score)


def fail_tracking(error_message: str):
    """Convenience function to mark tracking as failed."""
    tracker.fail_optimization(error_message)


def clear_tracking():
    """Convenience function to clear tracking."""
    tracker.clear_progress()


if __name__ == "__main__":
    # Test the progress tracker
    print("=== Testing Progress Tracker ===")
    
    # Simulate optimization
    start_tracking(5, "Starting test optimization")
    time.sleep(1)
    
    update_trial_progress(1, 0.75, "Testing first trial")
    time.sleep(1)
    
    update_trial_progress(2, 0.82, "Testing second trial")  
    time.sleep(1)
    
    update_trial_progress(3, 0.79, "Testing third trial")
    time.sleep(1)
    
    complete_tracking(0.85)
    
    print("✅ Progress tracker test completed!")