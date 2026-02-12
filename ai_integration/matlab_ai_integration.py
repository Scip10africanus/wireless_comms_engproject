"""
MATLAB-Python Integration for Real-Time AI Control
Bold (Unubold) - EEME30003 Major Project 1
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from feedback_agent import AdaptiveModulationAgent

# MATLAB Engine import (with error handling)
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("WARNING: MATLAB Engine not found. Please install it:")
    print("  In MATLAB: cd (fullfile(matlabroot,'extern','engines','python'))")
    print("             system('python setup.py install')")


class MATLABBridge:
    """Bridge between Python AI and MATLAB simulation."""
    
    def __init__(self, model_path='feedback_agent_model.pth', matlab_workspace='../'):
        
        print("="*70)
        print(" Initializing MATLAB-Python Bridge")
        print(" AI-Driven Adaptive Modulation Control")
        print("="*70)
        
        if not MATLAB_AVAILABLE:
            raise ImportError("MATLAB Engine not available")
        
        # Start MATLAB
        print("\n[1/3] Starting MATLAB engine...")
        try:
            self.eng = matlab.engine.start_matlab()
            print("  ✓ MATLAB engine started")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            raise
        
        # Setup paths
        print("\n[2/3] Setting up MATLAB workspace...")
        self.eng.addpath(matlab_workspace, nargout=0)
        self.eng.addpath(f"{matlab_workspace}tx", nargout=0)
        self.eng.addpath(f"{matlab_workspace}rx", nargout=0)
        self.eng.addpath(f"{matlab_workspace}channel", nargout=0)
        print("  ✓ MATLAB paths configured")
        
        # Load AI
        print("\n[3/3] Loading trained AI agent...")
        self.agent = AdaptiveModulationAgent()
        self.agent.load(model_path)
        print("  ✓ AI agent loaded")
        
        print(f"\n{'='*70}")
        print(" System Ready!")
        print(f"{'='*70}\n")
    
    def matlab_to_python_state(self, feedback, current_params):
        """Convert MATLAB feedback to Python state."""
        snr = feedback.get('estimatedSNR', 15.0) if isinstance(feedback, dict) else 15.0
        
        base_evm = 0.0
        enh_evm = 0.0
        if isinstance(feedback, dict) and 'bandAnalysis' in feedback:
            ba = feedback['bandAnalysis']
            if 'critical' in ba and 'evm' in ba['critical']:
                base_evm = ba['critical']['evm']
            if 'noncritical' in ba and 'evm' in ba['noncritical']:
                enh_evm = ba['noncritical']['evm']
        
        ssim = feedback.get('ssim', 0.8) if isinstance(feedback, dict) else 0.8
        psnr = feedback.get('psnr', 30.0) if isinstance(feedback, dict) else 30.0
        
        base_mod = current_params.get('baseMod', 4) if isinstance(current_params, dict) else 4
        enh_mod = current_params.get('enhancementMod', 64) if isinstance(current_params, dict) else 64
        
        if np.isnan(base_evm): base_evm = 0.0
        if np.isnan(enh_evm): enh_evm = 0.0
        if np.isnan(ssim): ssim = 0.8
        if np.isnan(psnr): psnr = 30.0
        
        state = np.array([snr, base_evm, enh_evm, base_mod, enh_mod, ssim, psnr])
        return state
    
    def python_to_matlab_params(self, action, current_params):
        """Convert Python action to MATLAB params."""
        base_mod, enh_mod = action
        
        new_params = current_params.copy() if isinstance(current_params, dict) else {}
        new_params['baseMod'] = float(base_mod)
        new_params['enhancementMod'] = float(enh_mod)
        
        return new_params
    
    def run_ai_controlled_simulation(self, n_frames=20, snr_profile=None):
        """Run closed-loop simulation with AI controller."""
        
        print("="*70)
        print(" Running AI-Controlled Closed-Loop Simulation")
        print("="*70)
        
        if snr_profile is None:
            snr_profile = [25, 22, 18, 15, 12, 10, 8, 6, 8, 10,
                          12, 15, 18, 22, 25, 28, 28, 30, 30, 30]
        
        if len(snr_profile) != n_frames:
            snr_profile = snr_profile[:n_frames]
        
        print("\n[1/3] Initializing MATLAB simulation...")
        params_matlab = self.eng.config_two_layer()
        params_matlab['verbose'] = False
        
        self.eng.eval("testImage = uint8(randi([0, 255], 64, 64));", nargout=0)
        test_image = self.eng.workspace['testImage']
        params_matlab['refImage'] = test_image
        
        print("  ✓ Test image: 64x64")
        print(f"  ✓ SNR: {min(snr_profile):.0f} - {max(snr_profile):.0f} dB")
        
        results = {
            'snr': snr_profile,
            'base_mod': [],
            'enh_mod': [],
            'ssim': [],
            'psnr': [],
            'ai_decisions': []
        }
        
        current_params = {
            'baseMod': params_matlab['baseMod'],
            'enhancementMod': params_matlab['enhancementMod']
        }
        
        print("\n[2/3] Running simulation...")
        print(f"\n{'Frame':>5} | {'SNR(dB)':>7} | {'Base':>8} | {'Enh':>7} | "
              f"{'SSIM':>6} | {'PSNR':>6} | {'AI Action'}")
        print("-" * 70)
        
        mod_names = {2: 'BPSK', 4: 'QPSK', 16: '16-QAM', 64: '64-QAM', 256: '256-QAM'}
        
        for frame in range(n_frames):
            # TX
            tx_signal, tx_info = self.eng.two_layer_tx_pipeline(
                test_image, params_matlab, nargout=2
            )
            
            # Channel
            channel_params = {
                'channelType': 'AWGN',
                'SNR_dB': float(snr_profile[frame])
            }
            rx_signal, _ = self.eng.channel_model(tx_signal, channel_params, nargout=2)
            
            # RX
            rx_image, feedback, rx_results = self.eng.two_layer_rx_pipeline(
                rx_signal, params_matlab, nargout=3
            )
            
            feedback_dict = {}
            if hasattr(feedback, '_fieldnames'):
                for field in feedback._fieldnames:
                    feedback_dict[field] = getattr(feedback, field)
            
            ssim_val = float(feedback_dict.get('ssim', 0.8))
            psnr_val = float(feedback_dict.get('psnr', 30.0))
            
            results['base_mod'].append(current_params['baseMod'])
            results['enh_mod'].append(current_params['enhancementMod'])
            results['ssim'].append(ssim_val)
            results['psnr'].append(psnr_val)
            
            # AI decision
            state = self.matlab_to_python_state(feedback_dict, current_params)
            state_normalized = self.agent.normalize_state(state)
            action_idx, action = self.agent.select_action(state_normalized, training=False)
            
            ai_decision = f"{mod_names[action[0]]} / {mod_names[action[1]]}"
            results['ai_decisions'].append(ai_decision)
            
            print(f"{frame+1:5d} | {snr_profile[frame]:7.0f} | "
                  f"{mod_names.get(int(current_params['baseMod']), 'N/A'):>8} | "
                  f"{mod_names.get(int(current_params['enhancementMod']), 'N/A'):>7} | "
                  f"{ssim_val:6.4f} | {psnr_val:6.2f} | {ai_decision}")
            
            # Update
            current_params = self.python_to_matlab_params(action, current_params)
            params_matlab['baseMod'] = current_params['baseMod']
            params_matlab['enhancementMod'] = current_params['enhancementMod']
        
        print("\n[3/3] Simulation complete!")
        print(f"  ✓ Average SSIM: {np.mean(results['ssim']):.3f}")
        print(f"  ✓ Average PSNR: {np.mean(results['psnr']):.1f} dB")
        
        self.plot_results(results, snr_profile)
        
        return results
    
    def plot_results(self, results, snr_profile):
        """Plot AI performance."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        frames = range(1, len(snr_profile) + 1)
        
        # SNR
        axes[0, 0].plot(frames, snr_profile, 'k-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('SNR (dB)')
        axes[0, 0].set_title('Channel Condition')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Modulation
        axes[0, 1].semilogy(frames, results['base_mod'], 'b-s', 
                           linewidth=2, markersize=6, label='Base')
        axes[0, 1].semilogy(frames, results['enh_mod'], 'r-o', 
                           linewidth=2, markersize=6, label='Enhancement')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Modulation Order')
        axes[0, 1].set_title('AI Modulation Adaptation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SSIM
        axes[1, 0].plot(frames, results['ssim'], 'g-o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=0.9, color='r', linestyle='--', label='Target', linewidth=1.5)
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Image Quality (SSIM)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # PSNR
        axes[1, 1].plot(frames, results['psnr'], 'm-o', linewidth=2, markersize=6)
        axes[1, 1].axhline(y=30, color='r', linestyle='--', label='Target', linewidth=1.5)
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('PSNR (dB)')
        axes[1, 1].set_title('Image Quality (PSNR)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('AI-Driven Adaptive Modulation - Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'ai_performance_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Plot saved: {filename}")
        plt.show()
    
    def shutdown(self):
        """Shutdown MATLAB."""
        print("\nShutting down MATLAB...")
        self.eng.quit()
        print("  ✓ MATLAB stopped")


def main():
    """Main demo."""
    bridge = MATLABBridge(
        model_path='feedback_agent_model.pth',
        matlab_workspace='../'
    )
    
    results = bridge.run_ai_controlled_simulation(n_frames=20)
    
    bridge.shutdown()
    
    print("\n" + "="*70)
    print(" AI-Driven Adaptive Modulation Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
    