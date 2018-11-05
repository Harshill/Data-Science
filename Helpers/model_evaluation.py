import pandas as pd
import seaborn as sns
import matplotlib.pyplot as graph
from rosey.graphing import plot_barplot
from sklearn.metrics import (accuracy_score, f1_score, 
                             roc_auc_score, roc_curve,
                             mean_squared_error, confusion_matrix)


def evaluate(model, x, true_y, classification=True):
    pred_y = model.predict(x)
    
    if classification:
        print(f'Accuracy score: {accuracy_score(true_y, pred_y)}')
        print(f'F1 score:       {f1_score(true_y, pred_y)}')
        print(f'ROC AUC score:  {roc_auc_score(true_y, pred_y)}')
    else:
        print(f'R2: {model.score(x, true_y)}')
        print(f'RMSE: {np.sqrt(mean_squared_error(true_y, pred_y))}')
        
        
def plot_roc(model, x, y, show_graph=True):
    prediction = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, prediction)
    
    graph.plot(fpr, tpr)
    graph.plot([0, 1], [0, 1])
    
    if show_graph:
        graph.show()
    
    
def plot_class_proba(model, x, y, show_graph=True, label_encoder=None):
    if label_encoder is not None:
        y = label_encoder.inverse_transform(y)
        
    df = pd.DataFrame({'Class': y, 'Probability': model.predict_proba(x)[:, 1]})
    sns.boxenplot(x='Class', y='Probability', data=df)
    
    if show_graph:
        graph.show()

        
def plot_confusion_matrix(model, x, true_y, show_graph=True, labels=None):
    pred_y = model.predict(x)
    
    if labels is None:
        conf_mat = pd.DataFrame(confusion_matrix(true_y, pred_y))
    else: 
        conf_mat = pd.DataFrame(confusion_matrix(true_y, pred_y), columns=labels, index=labels)
    sns.heatmap(conf_mat, annot=True, fmt='g')
    
    if show_graph:
        graph.show()
        
        
def plot_feature_importance(feature_names, feature_importances, show_graph=True):  
    plot_barplot(dict(zip(feature_names, feature_importances)), show_graph=show_graph)