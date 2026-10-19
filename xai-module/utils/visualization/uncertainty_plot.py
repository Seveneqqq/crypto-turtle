import matplotlib.pyplot as plt


def plot_masking_intervals(report, path):
    rows=[r for r in report['intervals'] if r['metric']=='top_minus_random']
    fig,ax=plt.subplots(figsize=(8,4))
    for i,row in enumerate(rows):
        lower,upper=row['ci95']
        ax.plot([lower,upper],[i,i],color='steelblue',linewidth=3)
        ax.plot(row['estimate'],i,'o',color='black')
    ax.axvline(0,color='gray',linestyle='--')
    ax.set_yticks(range(len(rows)),[r['method'].upper() for r in rows])
    ax.set_xlabel('Różnica spadku jakości: najważniejsze cechy minus losowe')
    ax.set_title('Maskowanie: oszacowania i 95% przedziały bootstrapowe')
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)
