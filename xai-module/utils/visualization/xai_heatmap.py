import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class XaiHeatmap:
    
    @staticmethod
    def plot(comparisons, save_path):
    
        data = []
    
        for key, value in comparisons.items():
        
            m1, m2 = key.split("_vs_")
    
            data.append({
                "method1": m1,
                "method2": m2,
                "spearman": value["spearman"]
            })
    
        df = pd.DataFrame(data)
    
        pivot = df.pivot(index="method1", columns="method2", values="spearman")
    
        plt.figure(figsize=(6, 5))
    
        sns.heatmap(
            pivot,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )
    
        plt.title("Agreement Between XAI Methods")
    
        plt.tight_layout()
    
        plt.savefig(save_path)
    
        plt.close()
    
    