import matplotlib.pyplot as plt

def mAP_plot(recalls, precisions):

    plt.plot(recalls.tolist(), precisions.tolist())
    plt.title('PR-Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()