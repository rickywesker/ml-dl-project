import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error

palette = ["#0e0e0e", 
           "#919191", 
           "#FFFFFF", 
           "#B5EEED", 
           "#0B90A7"]

params = {
    "rc": {
        "axes.facecolor": palette[2],
        "axes.grid": True,
        "grid.color": palette[1],
        "axes.edgecolor": palette[0],
        "figure.facecolor": palette[2],
        "axes.labelcolor": palette[0],
        "xtick.color": palette[0],
        "ytick.color": palette[0],
        "legend.labelcolor": palette[0],
        "legend.edgecolor": palette[2],
        "xtick.labelsize": 14,
        "ytick.labelsize":14,
        "font.size": 18,
        "font.family":"serif",
        "font.weight": "bold",

    }
}

sns.set(**params)

sns.despine()
sns.palplot(palette)


class CFG:
    # General
    SEED = 8615
    TARGET = 'Label'
    ID = 'id'
    
    # Training settings
    TEST_SIZE = 0.25
    SPLITS = 15
    SHUFFLE = True
    
    # Paths
    TRAIN = '../data/train.csv'
    TEST = './data/validation.csv'
    SUB = '/kaggle/input/playground-series-s4e4/sample_submission.csv'
    
    # Preprocessing settings
    OHE = False
    IMPUTER_MODE = None
    IQR_THRESHOLD = 3
    USE_ORIGINAL = False
    FEATURE_ENGINEERING = False
    
    # Plots
    BOXPLOT = {
                'boxprops': {
                    'edgecolor': '#0e0e0e',
                    'alpha': 1
                },
                'medianprops': {
                    'color': '#0e0e0e',
                    'linestyle': '-',
                    'linewidth': 0.7
                },
                'whiskerprops': {
                    'color': '#0e0e0e',
                    'linestyle': '--',
                    'linewidth': 0.7
                },
                'capprops': {
                    'color': '#0e0e0e',
                    'linewidth': 0.7
                },
                'flierprops': {
                    'marker': 'o',
                    'markersize': 4,
                    'markerfacecolor': '#fffdf7',
                    'markeredgecolor': '#0e0e0e',
                    'linewidth': 0.7
                }
            }

    TABLE = [
        {
            "selector": "tr",
            "props": [
                ("background-color", palette[-1]),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("font-size", "16pt"),
                ("color", palette[2]),
                ("text-align", "center"),
                ("font-family", "serif"),
            ],
        },
        {
            "selector": "th.blank.level0",
            "props": [
                ("background-color", palette[-1])
            ],
        },
        {
            "selector": "th.blank.level0:hover",
            "props": [
                ("background-color", "#E63946"),
            ],
        },
        {
            "selector": "th:hover",
            "props": [
                ("background-color", "#E63946"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
                ("font-size", "12pt"),
                ("font-family", "serif")
            ],
        },
        {
            "selector": "tr:hover td:hover",
            "props": [
                ("color", palette[2]),
                ("background-color", "#E63946"),
            ],
        }
    ]

class DataHandler:
    def __init__(self):
        self.train, self.test, self.sub = DataHandler.load_data([CFG.TRAIN, 
                                                                 CFG.TEST, 
                                                                 CFG.SUB])
        
        self.total = pd.concat([self.train.drop(CFG.TARGET, axis=1), self.test])
        self.total.reset_index(inplace=True, drop=True)
        
        self.viz = Visualizer(self.train, self.test, self.sub, self.total)
    
    def get_data(self):
        return self.train.copy(), self.test.copy()
    
    @staticmethod
    def load_data(paths:list, params={}, id_col=None) -> list:
        dataframes = []
        for path in paths:
            dataframes.append(pd.read_csv(path, **params))
            dataframes[-1].drop(CFG.ID, axis=1, inplace=True)
        
        return dataframes

    
class Visualizer:
    def __init__(self, train, test, target, total):
        self.train = train
        self.test = test
        self.target = target
        self.total = total
    
    def plot_kde(self, cols=3, fill=True, figsize=(15, 8), alpha=.5, linewidth=.75, colors=["#0F4BBD", "#BD280F"], title=None, legpos=(.5, 1.03)) -> None:
        numeric_cols = self.train.select_dtypes(include='number').columns
        numeric_cols = numeric_cols.drop(CFG.TARGET)
        rows = np.ceil(len(numeric_cols) / cols).astype(int)

        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

        for i, x in enumerate(numeric_cols):
            current_ax = ax[i // cols, i % cols]
            current_ax.grid(alpha=alpha, axis='both')
            sns.kdeplot(x=x,
                        color=colors[0],
                        fill=fill,
                        linewidth=linewidth,
                        alpha=alpha,
                        data=self.train,
                        label='Train',
                        ax=current_ax)
            sns.kdeplot(x=x,
                        color=colors[1],
                        fill=fill,
                        linewidth=linewidth,
                        alpha=alpha,
                        data=self.test,
                        label='Test',
                        ax=current_ax)
            current_ax.set_xlabel(x)

        fig.legend(labels=['Train', 'Test'], loc='upper center', bbox_to_anchor=legpos, ncol=2)

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        if len(numeric_cols) < (rows * cols):
            for i in range(len(numeric_cols), rows * cols):
                fig.delaxes(ax.flatten()[i])

        fig.tight_layout()
        plt.show()
    
    def show_target_distribution(self) -> None:
        self.train[CFG.TARGET].value_counts().plot(kind='bar')
    
    @staticmethod
    def corr_matrix(matrix, figsize=(15,15), 
                    cmap=sns.diverging_palette(5, 200, sep=10, s=100, l=55, as_cmap=True), 
                    title='Correlation Matrix', 
                    mask=True, 
                    round_=2,
                    linewidths=0.1) -> None:
        if mask:
            mask = np.triu(np.ones_like(
                           matrix.corr(), 
                           dtype=bool))
    
        fig = plt.figure(figsize=figsize)
        ax = sns.heatmap(matrix,
                        annot=True,
                        cmap=cmap,
                        fmt=f'.{round_}f',
                        square=True,
                        mask=mask,
                        linewidths=linewidths,
                        linecolor=palette[2],
                        vmin=-1.1,
                        vmax=1.1)
        
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.show()
    
    def plot_matrices(self, 
                  figsize=(13,10), 
                  cmap=sns.diverging_palette(5, 200, sep=10, s=100, l=55, as_cmap=True), 
                  mask=True,
                  annot=False,
                  round_=2,
                  linewidths=0.1,
                  annot_fontsize=8) -> None:
        numeric_cols = self.total.select_dtypes(include='number').columns
        corr_tr = self.train[numeric_cols].corr()
        corr_tt = self.test[numeric_cols].corr()

        if mask:
            mask = np.triu(np.ones_like(corr_tr, dtype=bool))

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        sns.heatmap(
            corr_tr, 
            mask=mask, 
            cmap=cmap,
            annot=annot, 
            square=True, 
            cbar=False,
            fmt=f'.{round_}f',
            vmin=-1.1,
            vmax=1.1,
            linewidths=linewidths,
            linecolor=palette[2],
            annot_kws={"fontsize": annot_fontsize},
            ax=ax[0]
        );

        sns.heatmap(
            corr_tt, 
            mask=mask, 
            cmap=cmap,
            fmt=f'.{round_}f',
            annot=annot,
            square=True, 
            vmin=-1.1,
            vmax=1.1,
            cbar=False,
            linewidths=linewidths,
            linecolor=palette[2],
            annot_kws={"fontsize": annot_fontsize},
            ax=ax[1]
        );

        ax[0].set_title('Correlation Matrix (Train)', size=12, fontweight='bold')
        ax[1].set_title('Correlation Matrix (Test)', size=12, fontweight='bold')
        
        ax[0].xaxis.grid(False)
        ax[0].yaxis.grid(False)
        ax[1].xaxis.grid(False)
        ax[1].yaxis.grid(False)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_boxplot(df, 
                     target, 
                     cols=3, 
                     alpha=.8, 
                     palette='mako',
                     figsize=(15, 8), 
                     y_title=None, 
                     title=None) -> None:
        
        numeric_cols = df.select_dtypes(include='number').columns
        numeric_cols = numeric_cols.drop(CFG.TARGET)
        rows = np.ceil(len(numeric_cols) / cols).astype(int)
    
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

        yt_labels = []
        for i, x in enumerate(numeric_cols):
            current_ax = ax[i // cols, i % cols]
            current_ax.grid(alpha=alpha, axis='x') 
            sns.boxplot(data=df, y=target, ax=current_ax, x=x, orient='h', palette=palette, **CFG.BOXPLOT)
            current_ax.set_xlabel(x)

            if y_title and i % cols == 0:
                if len(yt_labels) == 0:
                    yt_labels = [label.get_text() for label in current_ax.get_yticklabels()]
                current_ax.set_ylabel(y_title)
                current_ax.set_yticklabels(yt_labels)
            else:
                current_ax.set_ylabel('')
                current_ax.set_yticklabels('')

        if len(numeric_cols) < (rows * cols):
            for i in range(len(numeric_cols), rows * cols):
                fig.delaxes(ax.flatten()[i])

        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold')

        fig.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_regplot(df, 
                     cols=3, 
                     figsize=(15, 8), 
                     alpha=.05, 
                     colors=["#0F4BBD", "#BD280F"], 
                     title=None) -> None:
        num_cols = df.select_dtypes(include='number').columns
        num_cols = num_cols.drop(CFG.TARGET)

        rows = np.ceil(len(num_cols)/cols).astype(int)
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

        for i, x in enumerate(num_cols):
            current_ax = ax[i // cols, i % cols]
            current_ax.grid(alpha=alpha, axis='both')
            corr = round(df[[x, CFG.TARGET]].corr().iloc[0, 1], 2)
            sns.regplot(data = df,
                        x = x,
                        y = CFG.TARGET,
                        ax = current_ax,
                        scatter_kws = {'alpha': alpha,
                                       'color' : colors[0]},
                        line_kws = {'color': colors[1],
                                    'linestyle' : '--'},
                        )

            current_ax.set_title(f'{x} (Corr : {corr})')
            current_ax.set_xlabel('')

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        if len(num_cols) < (rows * cols):
            for i in range(len(num_cols), rows * cols):
                fig.delaxes(ax.flatten()[i])

        fig.tight_layout()
        plt.show()
    
    @staticmethod
    def display_data_info(df, name) -> None:
        matrix = pd.DataFrame(index=df.columns)
        total_missing = df.isnull().sum().sum()

        matrix['Count'] = df.shape[0]
        matrix['#Null'] = df.isnull().sum()
        matrix['%Null'] = df.isnull().mean().round(3)
        matrix['Unique'] = df.nunique()

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        matrix.loc[numerical_cols, 'Mean'] = df[numerical_cols].mean()
        matrix.loc[numerical_cols, 'Std'] = df[numerical_cols].std()
        matrix.loc[numerical_cols, 'Min'] = df[numerical_cols].min()
        matrix.loc[numerical_cols, 'Q1'] = df[numerical_cols].quantile(0.25)
        matrix.loc[numerical_cols, 'Q2'] = df[numerical_cols].median()
        matrix.loc[numerical_cols, 'Q3'] = df[numerical_cols].quantile(0.75)
        matrix.loc[numerical_cols, 'Max'] = df[numerical_cols].max()

        display(matrix.sort_values(['Mean']).style.set_table_styles(CFG.TABLE).background_gradient(cmap='Blues'))

        return f'\n{"-"*18}\n{name} Summary:\nTotal Entries: {df.shape[0]}\nTotal Features: {df.shape[1]}\nTotal Missing Values: {total_missing}\n{"-"*18}'
    
    def __str__(self) -> str:
        summary_train = ''
        summary_test = ''
        
        print('Train Dataset')
        summary_train += Visualizer.display_data_info(self.train, 'Train')
        
        print('Test Dataset')
        summary_test += Visualizer.display_data_info(self.test, 'Test')
        return summary_train + summary_test

class Processor:
    def  __init__(self, train, test, models):
        self.oof_train = pd.DataFrame()
        self.oof_test = pd.DataFrame()
        self.val_scores = pd.DataFrame()
        
        self.X = train.drop(CFG.TARGET, axis=1)
        self.y = train[CFG.TARGET]
        self.test = test
        
        self.models = models
        
    def process(self):
        for name, model in self.models:
            print(name)
            self.cross_val_score(name, model)
            print()
    
    def cross_val_score(self, name, model):
        oof_tr = np.zeros(self.X.shape[0])
        oof_cv = np.zeros((CFG.SPLITS, self.test.shape[0]))
        val_scores = []
        
        cv = StratifiedKFold(n_splits=CFG.SPLITS, 
                             shuffle=CFG.SHUFFLE,
                             random_state=CFG.SEED)
        
        for i, (train_index, test_index) in tqdm(enumerate(cv.split(self.X, self.y))):
            X_train, y_train = self.X.iloc[train_index], self.y.iloc[train_index]
            X_test, y_test = self.X.iloc[test_index], self.y.iloc[test_index]

            clf = model

            clf.fit(X_train, y_train)

            oof_tr[test_index] = np.clip(clf.predict(X_test), a_min=1, a_max=29)
            oof_cv[i, :] =  np.clip(clf.predict(self.test), a_min=1, a_max=29)
            
            rmsle_value = mean_squared_log_error(y_test, oof_tr[test_index], squared=False)
            val_scores.append(rmsle_value)
            
        print(f'Mean RMSLE: {np.array(val_scores).mean().round(4)} ± {np.array(val_scores).std().round(6)} \n')
        self.oof_train[name] = oof_tr
        self.oof_test[name] = oof_cv.mean(axis=0)
        self.val_scores[name] = val_scores
        
    @staticmethod
    def process_datasets(datasets, models):
        processors_oof_train = {k:[] for k in datasets}
        processors_oof_test = {k:[] for k in datasets}
        val_scores = {k:[] for k in datasets}

        for name, (tr, tt) in datasets.items():
            print(f"Creating processor: {name}")
            processor = Processor(tr, tt, models)
            processor.process()
            processors_oof_train[name] = processor.oof_train
            processors_oof_test[name]  = processor.oof_test
            val_scores[name] = processor.val_scores

        return processors_oof_train, processors_oof_test, val_scores
