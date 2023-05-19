import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, \
    confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import json

#helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    # if one_channel:
    #     img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
def write_batch_to_board(trainloader,grayscale, writer):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # create grid of images}
    print(images.shape)
    img_grid = torchvision.utils.make_grid(images)
    print(img_grid.shape)
    # show images
    matplotlib_imshow(img_grid, one_channel=False)
    # write to tensorboard
    writer.add_image('Random Training batch', img_grid)
#
# def plot_pr_to_board(test_probs, test_label, classes, writer):
#     # plot all the pr curves
#     for i in range(len(classes)):
#         add_pr_curve_tensorboard(i, test_probs, test_label, classes, writer)
#
# # helper function
# def add_pr_curve_tensorboard(class_index, test_probs, test_label,classes, writer):
#     '''
#     Takes in a "class_index" from 0 to 9 and plots the corresponding
#     precision-recall curve
#     '''
#
#
#     tensorboard_probs = np.asarray([probs[class_index] for probs in test_probs])
#     tensorboard_truth = np.asarray([label == class_index for label in test_label])
#
#     writer.add_pr_curve('PrecisionVsRecall/'+classes[class_index],
#                         tensorboard_truth,
#                         tensorboard_probs,
#                         global_step=0)
#     # writer.close()

def write_cm_to_board(probs, trues, classes, writer):
    preds = [np.argmax(prob, axis=-1) for prob in probs]
    cm = confusion_matrix(trues, preds)
    def plt_cm(cm):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes).plot(ax=ax)
        return fig

    writer.add_figure('Confusion Matrix', plt_cm(cm))

def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))

def start_tensor_board(args):
    writer = SummaryWriter(f'runs/{args.outname}')
    board_arguments = pretty_json(vars(args))
    writer.add_text("Training Arguments", board_arguments)
    return writer

def format_classification_report(report):
    # Split the report into lines
    lines = report.strip().split('\n')
    # Get the column names
    columns = [col.strip() for col in lines[0].split()]
    # Initialize an empty list to store the rows
    rows = []
    # Iterate over the remaining lines and parse the data
    for line in lines[2:]:
        row = [col.strip() for col in line.split()]
        # If the row is not empty, append it to the list of rows
        if row:
            rows.append(row)
    # Build the markdown table
    table = "| " + " | ".join(columns) + " |\n"
    table += "|-" + "-|-".join(['' for _ in columns]) + "|\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    # Return the markdown table
    return table

def write_report_to_board(probs, trues, classes, writer):
    preds = [np.argmax(prob, axis=-1) for prob in probs]
    report = classification_report(trues, preds,
                                   target_names=classes)
    writer.add_text("Classification report: \n ",
                    format_classification_report("Class "+ report))


def write_pr_to_board(probs, trues, classes, writer):
    probs = np.asarray(probs)
    trues = np.asarray(trues)

    true_encoded = np.zeros((trues.size, trues.max() + 1))
    true_encoded[np.arange(trues.size), trues] = 1

    pr_plot = plot_pr_curve(probs, true_encoded, classes)
    writer.add_figure('Precision Vs Recall', pr_plot)


def plot_pr_curve(pred_classes, true_classes, class_labels):
    # For each class
    n_classes = np.shape(pred_classes)[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):

        precision[i], recall[i], _ = precision_recall_curve(true_classes[:, i],
                                                            pred_classes[:, i])
        average_precision[i] = average_precision_score(true_classes[:, i], pred_classes[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(true_classes.ravel(),
                                                                    pred_classes.ravel())
    average_precision["micro"] = average_precision_score(true_classes, pred_classes,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    colors = ['crimson', 'green', 'purple', 'yellow', 'blue', 'fuchsia', 'gray']

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='black', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class %s (area = %0.2f)' % (class_labels[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    # plt.show()
    return fig

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    n_classes = np.shape(gt)[-1]
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_classes):
        AUROCs.append(roc_auc_score(gt_np, pred_np))
    return AUROCs


def plot_roc_curve(pred_classes, true_classes, class_labels):
    # calculate ROC curve per class
    n_classes = np.shape(pred_classes)[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_classes[:, i], pred_classes[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_classes.ravel(), pred_classes.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    lw = 2
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle=':', linewidth=4)

    colors = ['crimson', 'green', 'purple', 'yellow', 'blue', 'fuchsia', 'gray']
    for i, color, label in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC for class {2} (area = {1:0.2f})'.format(i, roc_auc[i], label))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    legend = np.insert(class_labels, 0, ['micro', 'macro'])
    plt.legend(loc="lower right")
    # plt.show()
    return fig

def write_roc_to_board(probs, trues, classes, writer):
    probs = np.asarray(probs)
    trues = np.asarray(trues)

    true_encoded = np.zeros((trues.size, trues.max() + 1))
    true_encoded[np.arange(trues.size), trues] = 1

    pr_plot = plot_roc_curve(probs, true_encoded, classes)
    writer.add_figure('ROC Curve', pr_plot)
