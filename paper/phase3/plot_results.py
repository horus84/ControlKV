import matplotlib.pyplot as plt
import csv

def plot_degradation():
    labels = ['Dynamic (16-bit)', 'HQQ 8-bit', 'HQQ 4-bit']
    valid_rates = [1.0, 1.0, 0.0]
    coherent_rates = [0.93, 0.93, 0.50]
    
    x = range(len(labels))
    plt.bar([i - 0.2 for i in x], valid_rates, width=0.4, label='Tool Valid Rate', color='blue')
    plt.bar([i + 0.2 for i in x], coherent_rates, width=0.4, label='Coherent Rate', color='orange')
    plt.xticks(x, labels)
    plt.ylabel('Rate')
    plt.title('Reliability Cliff in Qwen2.5-1.5B (Ctx: 1024)')
    plt.legend()
    plt.savefig('paper/phase3/reliability_cliff.png')

if __name__ == '__main__':
    plot_degradation()
