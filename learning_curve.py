import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def plot_learning_curve(estimator, X, y, cv, modelname="Model", scoring='recall_macro'):
    pipeline = make_pipeline(
        StandardScaler(),  
        estimator
    )
    train_sizes, train_scores, test_scores = learning_curve( pipeline, X, y, cv=cv, train_sizes=np.linspace(.1, 1.0, 5),
        scoring=scoring,shuffle=True, random_state=42,n_jobs=-1  )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.title(f'Learning Curve\n{modelname} (Macro Recall)', fontsize=14)
    plt.xlabel('Training examples', fontsize=12)
    plt.ylabel('Macro Recall', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot shaded areas
    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,alpha=0.2, color='#1f77b4')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,alpha=0.2, color='#ff7f0e')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='#1f77b4', 
            linewidth=2, markersize=8, label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='#ff7f0e', 
            linewidth=2, markersize=8, label='Cross-Val Score')

    best_test_score = test_scores_mean.max()
    best_idx = np.argmax(test_scores_mean)
    plt.annotate(f'Best: {best_test_score:.3f}', xy=(train_sizes[best_idx], best_test_score),
                xytext=(10, 10), textcoords='offset points',bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->'))

    plt.legend(loc='lower right', fontsize=10)
    plt.ylim([0, 1.1])

    
    plt.show()
