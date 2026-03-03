"""
ML Closed-Loop Bridge - Python DQN integration with new TX pipeline

Extends the existing feedback_agent.py DQN agent to work with the new
TX/RX pipeline and PolicyController CSV-based feedback loop.

Usage:
    From MATLAB:
        pyrunfile("ml_closed_loop_bridge.py", "run_bridge")
    Or standalone:
        python ml_closed_loop_bridge.py
"""

import numpy as np
import csv
import os
import time
from datetime import datetime

# Import existing DQN agent
try:
    from feedback_agent import AdaptiveModulationAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("WARNING: feedback_agent.py not found or failed to import.")
    print("  DQN agent will not be available. Using rule-based fallback.")


class ClosedLoopBridge:
    """Bridge between MATLAB closed-loop simulation and Python DQN agent."""

    # MCS to modulation order mapping (802.11a)
    MCS_TO_MOD = {0: 2, 1: 2, 2: 4, 3: 4, 4: 16, 5: 16, 6: 64, 7: 64}
    MOD_TO_MCS_BASE = {2: 0, 4: 2, 16: 4}  # Base layer MCS lookup
    MOD_TO_MCS_ENH = {4: 2, 16: 4, 64: 6, 256: 7}  # Enhancement MCS lookup

    def __init__(self, policy_csv_path='policy.csv', model_path='feedback_agent_model.pth'):
        self.policy_csv_path = policy_csv_path
        self.model_path = model_path

        # Channel history tracking
        self.snr_history = []
        self.ssim_history = []
        self.psnr_history = []
        self.action_history = []

        # Load DQN agent if available
        self.agent = None
        if DQN_AVAILABLE:
            try:
                self.agent = AdaptiveModulationAgent()
                if os.path.exists(model_path):
                    self.agent.load(model_path)
                    print(f"[ML Bridge] DQN agent loaded from {model_path}")
                else:
                    print(f"[ML Bridge] No trained model found at {model_path}")
                    print(f"[ML Bridge] Using untrained DQN (will explore randomly)")
            except Exception as e:
                print(f"[ML Bridge] Failed to initialize DQN agent: {e}")
                self.agent = None

        if self.agent is None:
            print("[ML Bridge] Using rule-based fallback controller")

    def update_from_feedback(self, feedback_dict):
        """
        Process RX feedback and update policy CSV.

        Args:
            feedback_dict: dict with keys matching RX v2 feedback struct:
                - ssim, psnr, hasQualityMetrics, meetsQualityTargets
                - baseLayerStatus, detailLayerStatus
                - baseLayerError, detailLayerError
                - needsUrgentChange, urgentReason
                - predictedSNR, predictionConfidence, channelTrend
                - snr_dB (estimated/measured SNR)
                - current_K, current_Qb_idx, current_Qe_idx
                - current_mcs_base, current_mcs_enh

        Returns:
            dict with recommended policy values
        """
        # Update history
        snr = feedback_dict.get('snr_dB', 20.0)
        ssim = feedback_dict.get('ssim', 0.8)
        psnr = feedback_dict.get('psnr', 30.0)

        self.snr_history.append(snr)
        self.ssim_history.append(ssim)
        self.psnr_history.append(psnr)

        # Get current params
        current_mcs_base = feedback_dict.get('current_mcs_base', 0)
        current_mcs_enh = feedback_dict.get('current_mcs_enh', 5)
        current_K = feedback_dict.get('current_K', 20)
        current_Qb = feedback_dict.get('current_Qb_idx', 0)
        current_Qe = feedback_dict.get('current_Qe_idx', 0)

        # Convert MCS to modulation orders for DQN state
        base_mod = self.MCS_TO_MOD.get(current_mcs_base, 4)
        enh_mod = self.MCS_TO_MOD.get(current_mcs_enh, 64)

        if self.agent is not None:
            # Use DQN agent
            recommendation = self._dqn_recommend(feedback_dict, base_mod, enh_mod)
        else:
            # Rule-based fallback
            recommendation = self._rule_based_recommend(feedback_dict, current_mcs_base, current_mcs_enh)

        # Build full policy update
        policy = {
            'version': len(self.snr_history),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tx_mode': 'split',
            'K': recommendation.get('K', current_K),
            'Qb_idx': recommendation.get('Qb_idx', current_Qb),
            'Qe_idx': recommendation.get('Qe_idx', current_Qe),
            'mcs_base': recommendation.get('mcs_base', current_mcs_base),
            'mcs_enh': recommendation.get('mcs_enh', current_mcs_enh),
            'crc_flag': 1,
            'endian_flag': 1,
            'codec_id': 0,
            'reserved_flag': 0,
            'mux_gap_us': 0,
        }

        # Write to CSV
        self._write_policy_csv(policy)

        return policy

    def _dqn_recommend(self, feedback, base_mod, enh_mod):
        """Get DQN agent recommendation."""
        # Build state vector matching feedback_agent.py format
        snr = feedback.get('snr_dB', 20.0)
        base_evm = feedback.get('baseLayerError', 10.0)
        enh_evm = feedback.get('detailLayerError', 20.0)
        ssim = feedback.get('ssim', 0.8)
        psnr = feedback.get('psnr', 30.0)

        # Clean NaNs
        if np.isnan(base_evm): base_evm = 10.0
        if np.isnan(enh_evm): enh_evm = 20.0
        if np.isnan(ssim): ssim = 0.8
        if np.isnan(psnr): psnr = 30.0

        state = np.array([snr, base_evm, enh_evm, base_mod, enh_mod, ssim, psnr])
        state_norm = self.agent.normalize_state(state)

        # Select action (exploit mode - no exploration)
        action_idx, (rec_base_mod, rec_enh_mod) = self.agent.select_action(state_norm, training=False)

        self.action_history.append(action_idx)

        # Online learning: if we have a previous transition, train
        if len(self.ssim_history) >= 2:
            prev_state = np.array([
                self.snr_history[-2],
                feedback.get('baseLayerError', 10.0),
                feedback.get('detailLayerError', 20.0),
                base_mod, enh_mod,
                self.ssim_history[-2],
                self.psnr_history[-2]
            ])
            prev_state_norm = self.agent.normalize_state(prev_state)

            prev_action = self.action_history[-2] if len(self.action_history) >= 2 else action_idx
            prev_action_tuple = self.agent.actions[prev_action]

            reward = self.agent.calculate_reward(
                ssim, psnr,
                feedback.get('needsUrgentChange', False),
                prev_action_tuple,
                (rec_base_mod, rec_enh_mod)
            )

            self.agent.memory.push(prev_state_norm, prev_action, reward, state_norm, False)
            self.agent.train_step(batch_size=32)

        # Convert modulation orders back to MCS
        rec_mcs_base = self.MOD_TO_MCS_BASE.get(rec_base_mod, 0)
        rec_mcs_enh = self.MOD_TO_MCS_ENH.get(rec_enh_mod, 5)

        result = {
            'mcs_base': rec_mcs_base,
            'mcs_enh': rec_mcs_enh,
        }

        # Also recommend K based on quality
        current_K = feedback.get('current_K', 20)
        if ssim < 0.6:
            result['K'] = min(60, current_K + 5)
        elif ssim > 0.95:
            result['K'] = max(5, current_K - 2)
        else:
            result['K'] = current_K

        return result

    def _rule_based_recommend(self, feedback, current_mcs_base, current_mcs_enh):
        """Rule-based fallback when DQN is not available."""
        ssim = feedback.get('ssim', 0.8)
        psnr = feedback.get('psnr', 30.0)
        urgent = feedback.get('needsUrgentChange', False)
        current_K = feedback.get('current_K', 20)

        new_mcs_base = current_mcs_base
        new_mcs_enh = current_mcs_enh
        new_K = current_K

        if urgent:
            new_mcs_base = max(0, current_mcs_base - 2)
            new_mcs_enh = max(0, current_mcs_enh - 2)
            new_K = min(60, current_K + 5)
        elif ssim < 0.7:
            new_mcs_base = max(0, current_mcs_base - 1)
            new_mcs_enh = max(0, current_mcs_enh - 1)
            new_K = min(60, current_K + 2)
        elif ssim > 0.95 and psnr > 35:
            new_mcs_enh = min(7, current_mcs_enh + 1)
            new_K = max(5, current_K - 1)

        # Channel prediction influence
        predicted_snr = feedback.get('predictedSNR', None)
        if predicted_snr is not None and not np.isnan(predicted_snr):
            current_snr = feedback.get('snr_dB', 20.0)
            if predicted_snr < current_snr - 5:
                new_mcs_enh = max(0, new_mcs_enh - 1)

        return {
            'mcs_base': new_mcs_base,
            'mcs_enh': new_mcs_enh,
            'K': new_K,
        }

    def _write_policy_csv(self, policy):
        """Write policy dict to CSV for PolicyController."""
        fields = ['version', 'timestamp', 'tx_mode', 'K', 'Qb_idx', 'Qe_idx',
                  'crc_flag', 'endian_flag', 'codec_id', 'reserved_flag',
                  'mcs_base', 'mcs_enh', 'mux_gap_us']

        with open(self.policy_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(policy)

    def save_agent(self, path=None):
        """Save the DQN agent (with online learning updates)."""
        if self.agent is not None:
            save_path = path or self.model_path
            self.agent.save(save_path)
            print(f"[ML Bridge] Agent saved to {save_path}")

    def get_stats(self):
        """Return performance statistics."""
        stats = {
            'n_frames': len(self.snr_history),
            'avg_ssim': np.mean(self.ssim_history) if self.ssim_history else 0,
            'avg_psnr': np.mean(self.psnr_history) if self.psnr_history else 0,
            'snr_range': (min(self.snr_history), max(self.snr_history)) if self.snr_history else (0, 0),
            'agent_type': 'DQN' if self.agent else 'rule-based',
        }
        return stats


def run_bridge(policy_csv_path='policy.csv', model_path='feedback_agent_model.pth'):
    """Entry point for MATLAB pyrunfile calls."""
    bridge = ClosedLoopBridge(policy_csv_path, model_path)
    return bridge


def demo():
    """Standalone demo simulating feedback loop."""
    print("=" * 70)
    print("  ML Closed-Loop Bridge - Demo")
    print("=" * 70)

    bridge = ClosedLoopBridge()

    # Simulate 10 frames of feedback
    snr_profile = [30, 25, 20, 15, 10, 12, 18, 22, 28, 30]

    for i, snr in enumerate(snr_profile):
        # Simulate feedback from RX
        ssim = max(0.3, min(1.0, 0.5 + snr / 60.0 + np.random.randn() * 0.05))
        psnr = max(10, min(50, snr + 5 + np.random.randn() * 2))

        feedback = {
            'snr_dB': snr,
            'ssim': ssim,
            'psnr': psnr,
            'hasQualityMetrics': True,
            'meetsQualityTargets': ssim >= 0.9 and psnr >= 30,
            'baseLayerError': max(0, 30 - snr + np.random.randn() * 3),
            'detailLayerError': max(0, 40 - snr + np.random.randn() * 5),
            'needsUrgentChange': ssim < 0.5,
            'urgentReason': 'Low quality' if ssim < 0.5 else '',
            'predictedSNR': snr_profile[min(i+1, len(snr_profile)-1)],
            'predictionConfidence': 0.7,
            'current_K': 20,
            'current_mcs_base': 0,
            'current_mcs_enh': 5,
        }

        policy = bridge.update_from_feedback(feedback)

        print(f"Frame {i+1:2d} | SNR={snr:5.1f} dB | SSIM={ssim:.3f} | "
              f"mcsB={policy['mcs_base']} mcsE={policy['mcs_enh']} K={policy['K']}")

    stats = bridge.get_stats()
    print(f"\nStats: {stats}")
    bridge.save_agent()


if __name__ == "__main__":
    demo()
