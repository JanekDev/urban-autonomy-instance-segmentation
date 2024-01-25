import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


class AveragePrecision(torch.nn.Module):
    def __init__(self, num_classes, iou_threshold=None, calculate_full_ap=False):
        super(AveragePrecision, self).__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.calculate_full_ap = calculate_full_ap

        if calculate_full_ap:
            self.precisions = [[] for _ in range(num_classes)]
            self.recalls = [[] for _ in range(num_classes)]
        else:
            self.ap_scores = np.zeros(num_classes)

    def forward(self, predicted_masks, true_masks, iou_threshold=None):
        """
        Calculate the average precision for instance segmentation.

        Args:
            predicted_masks (torch.Tensor): Predicted instance masks (B, C, H, W)
            true_masks (torch.Tensor): True instance masks (B, C, H, W)
            iou_threshold (float, optional): IoU threshold for matching predictions to true instances.
                If not provided, the class-level IoU threshold will be used.

        Returns:
            float: Mean Average Precision (mAP) over all classes.
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        predicted_masks = predicted_masks.cpu().detach().numpy()
        true_masks = true_masks.cpu().detach().numpy()

        if self.calculate_full_ap:
            for class_idx in range(self.num_classes):
                y_true_class = true_masks[:, class_idx].reshape(-1)
                y_pred_class = predicted_masks[:, class_idx].reshape(-1)

                precision, recall, _ = precision_recall_curve(
                    y_true_class, y_pred_class
                )
                self.precisions[class_idx].append(precision)
                self.recalls[class_idx].append(recall)
        else:
            for class_idx in range(self.num_classes):
                y_true_class = true_masks[:, class_idx].reshape(-1)
                y_pred_class = predicted_masks[:, class_idx].reshape(-1)

                if iou_threshold is not None:
                    # Calculate IoU between predicted and true masks
                    intersection = np.sum(y_true_class * y_pred_class)
                    union = np.sum((y_true_class + y_pred_class) > 0)
                    iou = intersection / (union + 1e-7)

                    # Calculate AP with IoU threshold
                    if iou >= iou_threshold:
                        ap = average_precision_score(y_true_class, y_pred_class)
                        self.ap_scores[class_idx] += ap
                else:
                    # Calculate AP without IoU threshold
                    ap = average_precision_score(y_true_class, y_pred_class)
                    self.ap_scores[class_idx] += ap

        if not self.calculate_full_ap:
            mean_ap = np.mean(self.ap_scores)
            return mean_ap

    def get_precision_recall_curves(self):
        """
        Get precision-recall curves for each class.

        Returns:
            List of lists: A list of precision and recall curves for each class.
        """
        if self.calculate_full_ap:
            return self.precisions, self.recalls
        else:
            return None, None
