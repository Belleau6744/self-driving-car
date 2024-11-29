import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("experiment_results.csv")

# Analyze results
def plot_fitness_trends(df):
    plt.figure(figsize=(10, 6))
    for pop_size in df["Population Size"].unique():
        for mut_rate in df["Mutation Rate"].unique():
            subset = df[(df["Population Size"] == pop_size) & (df["Mutation Rate"] == mut_rate)]
            plt.plot(subset["Generation"], subset["Best Fitness"],
                     label=f"Pop: {pop_size}, Mut: {mut_rate}")

    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Trends by Population Size and Mutation Rate")
    plt.legend()
    plt.show()

def plot_heatmap(df):
    heatmap_data = df.groupby(["Population Size", "Mutation Rate"])["Best Fitness"].max().unstack()
    sns.heatmap(heatmap_data, annot=True, cmap="viridis")
    plt.title("Max Fitness by Population Size and Mutation Rate")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Population Size")
    plt.show()

# Run analysis
plot_fitness_trends(df)
plot_heatmap(df)
