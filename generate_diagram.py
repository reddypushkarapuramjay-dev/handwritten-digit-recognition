"""
Generate architecture diagram for the frontend.
Run this script to create the architecture diagram image.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

def generate_architecture_diagram(save_path='static/images/architecture_diagram.png'):
    """Generate a professional architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#10b981',
        'hidden1': '#2563eb',
        'hidden2': '#f59e0b',
        'output': '#ef4444',
        'arrow': '#64748b',
        'text': '#1e293b',
        'bg': '#f8fafc'
    }
    
    # Layer definitions
    layers = [
        {'name': 'Input Layer', 'size': 784, 'desc': '28×28 = 784 pixels\n(Flattened)', 'x': 1.5, 'color': colors['input'], 'neurons_shown': 8},
        {'name': 'Hidden Layer 1', 'size': 128, 'desc': '128 neurons\n+ ReLU + Dropout(0.2)', 'x': 4.5, 'color': colors['hidden1'], 'neurons_shown': 8},
        {'name': 'Hidden Layer 2', 'size': 64, 'desc': '64 neurons\n+ ReLU + Dropout(0.2)', 'x': 8.0, 'color': colors['hidden2'], 'neurons_shown': 8},
        {'name': 'Output Layer', 'size': 10, 'desc': '10 neurons (0-9)\n+ Softmax', 'x': 11.0, 'color': colors['output'], 'neurons_shown': 10},
    ]
    
    y_center = 4.5
    neuron_radius = 0.28
    neuron_spacing = 0.7
    
    # Draw layers
    for i, layer in enumerate(layers):
        # Background box for layer
        box_width = 2.2
        box_height = 6.5
        box_x = layer['x'] - box_width/2
        box_y = y_center - box_height/2
        
        # Layer background
        rect = plt.Rectangle((box_x, box_y), box_width, box_height,
                           facecolor=layer['color'], alpha=0.05,
                           edgecolor=layer['color'], linewidth=2, linestyle='--')
        ax.add_patch(rect)
        
        # Draw neurons
        n_shown = layer['neurons_shown']
        start_y = y_center + (n_shown - 1) * neuron_spacing / 2
        
        for j in range(n_shown):
            y_pos = start_y - j * neuron_spacing
            circle = plt.Circle((layer['x'], y_pos), neuron_radius,
                              facecolor=layer['color'], edgecolor='white',
                              linewidth=2, alpha=0.9, zorder=10)
            ax.add_patch(circle)
            
            # Add neuron index for output layer
            if i == 3:
                ax.text(layer['x'], y_pos, str(j), ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white', zorder=11)
        
        # Add ellipsis for hidden layers
        if layer['size'] > n_shown and i < 3:
            ax.text(layer['x'], y_center + 3.0, '⋮', ha='center', va='center',
                   fontsize=24, color=layer['color'], fontweight='bold', zorder=10)
        
        # Layer label
        ax.text(layer['x'], box_y - 0.6, layer['name'], ha='center', va='top',
               fontsize=13, fontweight='bold', color=layer['color'])
        ax.text(layer['x'], box_y - 1.1, layer['desc'], ha='center', va='top',
               fontsize=10, color=colors['text'], linespacing=1.5)
        
        # Parameter count
        if i == 0:
            param_text = f'784 × 128 + 128 = 100,480 params'
        elif i == 1:
            param_text = f'128 × 64 + 64 = 8,256 params'
        elif i == 2:
            param_text = f'64 × 10 + 10 = 650 params'
        else:
            param_text = ''
        
        if param_text:
            ax.text(layer['x'], box_y - 2.3, param_text, ha='center', va='top',
                   fontsize=9, color=colors['text'], alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=layer['color'], alpha=0.9))
    
    # Draw arrows between layers
    for i in range(len(layers) - 1):
        start_x = layers[i]['x'] + 1.1
        end_x = layers[i+1]['x'] - 1.1
        
        # Draw multiple connection lines to show dense connectivity
        for offset in [-2.0, -1.0, 0, 1.0, 2.0]:
            ax.annotate('', xy=(end_x, y_center + offset), xytext=(start_x, y_center + offset),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                                     alpha=0.3, lw=1.5, shrinkA=5, shrinkB=5),
                       zorder=5)
        
        # Main arrow with label
        mid_x = (start_x + end_x) / 2
        ax.annotate('', xy=(end_x, y_center), xytext=(start_x, y_center),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                                 alpha=0.8, lw=3, shrinkA=5, shrinkB=5),
                   zorder=5)
        
        # Weight matrix dimensions
        if i == 0:
            w_text = 'W₁: [128, 784]'
        elif i == 1:
            w_text = 'W₂: [64, 128]'
        else:
            w_text = 'W₃: [10, 64]'
        
        ax.text(mid_x, y_center + 2.8, w_text, ha='center', va='center',
               fontsize=10, fontweight='bold', color=colors['arrow'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['arrow'], alpha=0.9))
    
    # Title
    ax.text(7, 7.3, 'Neural Network Architecture', ha='center', va='center',
           fontsize=20, fontweight='bold', color=colors['text'])
    ax.text(7, 6.9, 'Fully Connected ANN for MNIST Digit Recognition', ha='center', va='center',
           fontsize=13, color=colors['text'], alpha=0.7)
    
    # Summary box
    summary_text = (
        "Architecture Summary\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "• Input: 784 features (28×28)\n"
        "• Hidden 1: 128 units + ReLU + Dropout\n"
        "• Hidden 2: 64 units + ReLU + Dropout\n"
        "• Output: 10 units (logits)\n"
        "• Loss: CrossEntropyLoss\n"
        "• Optimizer: Adam (lr=0.001)\n"
        "• Total Parameters: ~109,386\n"
        "• Test Accuracy: ~98%"
    )
    
    ax.text(13.2, 4.5, summary_text, ha='right', va='center',
           fontsize=10, color=colors['text'],
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', 
                     edgecolor=colors['arrow'], linewidth=1),
           fontfamily='monospace', linespacing=1.6)
    
    # Data flow annotation
    ax.annotate('Forward Pass →', xy=(10.5, 1.5), xytext=(3, 1.5),
               arrowprops=dict(arrowstyle='->', color=colors['primary'] if 'primary' in colors else colors['hidden1'], 
                             lw=3, alpha=0.5),
               fontsize=12, fontweight='bold', color=colors['hidden1'],
               ha='center', va='center')
    
    ax.text(7, 1.0, 'x → FC1 → ReLU → Dropout → FC2 → ReLU → Dropout → FC3 → Logits → Softmax → Probabilities',
           ha='center', va='center', fontsize=10, color=colors['text'], alpha=0.7,
           fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Architecture diagram saved to {save_path}")
    plt.close()


def generate_all_static_images():
    """Generate all static images needed for the frontend."""
    os.makedirs('static/images', exist_ok=True)
    
    print("Generating architecture diagram...")
    generate_architecture_diagram('static/images/architecture_diagram.png')
    
    print("Done!")


if __name__ == '__main__':
    generate_all_static_images()