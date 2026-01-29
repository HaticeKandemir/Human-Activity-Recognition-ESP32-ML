"""
Script Name: Final Evaluator & Reporting
=============================================================================
This script merges the results from the Classic ML and Deep Learning pipelines,
calculates a final ranking score, and generates summary visualizations.

Key Features:
    - **Merge:** Combines 'results_classic.csv' and 'results_deep.csv'.
    - **Ranking Score:** Calculates 'Accuracy / Latency' to find the most efficient model.
    - **Plot:** Generates an 'Accuracy vs. Latency' scatter plot to visualize
      the trade-off between performance and speed.
    - **Report:** Saves the final consolidated report to CSV.

Input: CSV files from the 'results/' directory.
Output: 'final_comparison_report.csv' and 'accuracy_vs_latency_plot.png'.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
OUTPUT_FILE = "final_comparison_report.csv"


def main():
    """
    Main execution driver for generating the final evaluation report.
    """
    print(">>> 5. FINAL EVALUATION STARTED")

    try:
        df_c = pd.read_csv(os.path.join(RESULTS_DIR, "results_classic.csv"))
        df_d = pd.read_csv(os.path.join(RESULTS_DIR, "results_deep.csv"))
    except FileNotFoundError:
        print("Error: Missing result files.")
        return

    df_all = pd.concat([df_c, df_d], ignore_index=True)

    # Calculate Score
    df_all["Ranking_Score"] = df_all["Accuracy"] / df_all["Inference_Latency_Sec"]
    df_all = df_all.sort_values(by="Accuracy", ascending=False)

    df_all.to_csv(os.path.join(RESULTS_DIR, OUTPUT_FILE), index=False)

    print("\nFINAL SUMMARY:")
    print(df_all[["Model", "Accuracy", "F1_Score", "Ranking_Score", "Best_Params"]])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_all,
        x="Inference_Latency_Sec",
        y="Accuracy",
        hue="Model",
        style="Model",
        s=200,
        palette="deep",
    )
    plt.xscale("log")
    plt.title("Accuracy vs. Latency Trade-off")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_latency_plot.png"))
    print("\nReport and Plots Saved.")


if __name__ == "__main__":
    main()
