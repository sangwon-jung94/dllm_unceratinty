import numpy as np
import matplotlib.pyplot as plt

def plot_uncertainty_over_time(uncertainty_history, save_path='uncertainty_over_timesteps.png', title_suffix=''):
    '''
    Plot uncertainty metrics over diffusion timesteps with mean ± std bands.
    
    Args:
        uncertainty_history: Dict containing timestep-wise uncertainty metrics
        save_path: Path to save the plot
        title_suffix: Additional text to add to the title
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    
    timesteps = uncertainty_history['timesteps']
    
    # Convert to numpy arrays for easier computation
    timesteps_arr = np.array(timesteps)
    mean_total = np.array(uncertainty_history['mean_total'])
    mean_epistemic = np.array(uncertainty_history['mean_epistemic'])
    mean_aleatoric = np.array(uncertainty_history['mean_aleatoric'])
    std_total = np.array(uncertainty_history['std_total'])
    std_epistemic = np.array(uncertainty_history['std_epistemic'])
    std_aleatoric = np.array(uncertainty_history['std_aleatoric'])
    
    # Plot Total Uncertainty with band
    ax.plot(timesteps, mean_total, 
            label='Total Uncertainty', linewidth=2.5, marker='o', markersize=4, 
            color='#2ca02c', alpha=0.9)
    ax.fill_between(timesteps, mean_total - std_total, mean_total + std_total,
                     alpha=0.2, color='#2ca02c')
    
    # Plot Epistemic Uncertainty with band
    ax.plot(timesteps, mean_epistemic, 
            label='Epistemic Uncertainty', linewidth=2.5, marker='s', markersize=4, 
            color='#1f77b4', alpha=0.9)
    ax.fill_between(timesteps, mean_epistemic - std_epistemic, mean_epistemic + std_epistemic,
                     alpha=0.2, color='#1f77b4')
    
    # Plot Aleatoric Uncertainty with band
    ax.plot(timesteps, mean_aleatoric, 
            label='Aleatoric Uncertainty', linewidth=2.5, marker='^', markersize=4, 
            color='#ff7f0e', alpha=0.9)
    ax.fill_between(timesteps, mean_aleatoric - std_aleatoric, mean_aleatoric + std_aleatoric,
                     alpha=0.2, color='#ff7f0e')
    
    ax.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax.set_ylabel('Uncertainty (Entropy)', fontsize=12)
    title = 'Uncertainty Over Diffusion Timesteps (Mean ± Std)\n(Averaged over all masked token positions)'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


# Load existing data and create new plot
data_path = './result/gsm8k_uncertainty_aware_epistemic/uncertainty_data.npz'
output_path = './result/gsm8k_uncertainty_aware_epistemic/uncertainty_over_timesteps_NEW.png'

print(f"Loading data from: {data_path}")
data = np.load(data_path)
uncertainty_history = {key: data[key].tolist() for key in data.files}

print(f"Data keys: {list(uncertainty_history.keys())}")
print(f"Number of timesteps: {len(uncertainty_history['timesteps'])}")

print("\nCreating new plot...")
plot_uncertainty_over_time(uncertainty_history, output_path, 'Test: New Band Plot')

print("\nDone! Check the output:")
print(f"  Old plot: ./result/gsm8k_uncertainty_aware_epistemic/uncertainty_over_timesteps.png")
print(f"  New plot: {output_path}")
