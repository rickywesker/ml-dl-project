import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_feature_dist(df: pd.DataFrame, test:pd.DataFrame, feature: str = "all"):
    """
    Plot the distribution of features in the dataset, with subplots in one canvas
    First row is the df, second row is the test with red color
    """
    if feature == "all":
        features = df.columns
    else:
        features = [feature]
    fig, axs = plt.subplots(2, len(features), figsize=(20, 10))
    #add a title
    fig.suptitle('Feature Distribution in Train and Test', fontsize=16)
    for i, f in enumerate(features):
        sns.histplot(df[f], ax=axs[0, i], kde=True)
        sns.histplot(test[f], ax=axs[1, i], kde=True, color='red')
        axs[0, i].set_title(f"Train {f}")
        axs[1, i].set_title(f"Test {f}")
    plt.show()

