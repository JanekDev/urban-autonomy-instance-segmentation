from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOEvaluator:
    def __init__(self, ground_truth_path, result_path):
        """
        Initialize the COCO Evaluator.
        :param ground_truth_path: Path to the COCO ground truth file.
        :param result_path: Path to the result file (annotations to be evaluated).
        """
        self.ground_truth_path = ground_truth_path
        self.result_path = result_path
        self.coco_gt = None
        self.coco_dt = None

    def load_data(self):
        """
        Loads the ground truth and result data using the COCO API.
        """
        self.coco_gt = COCO(self.ground_truth_path)
        self.coco_dt = self.coco_gt.loadRes(self.result_path)

    def evaluate(self, eval_type="bbox"):
        """
        Perform the COCO evaluation.
        :param eval_type: Type of evaluation (e.g., 'bbox', 'segm', 'keypoints').
        :return: None. Prints out the evaluation results.
        """
        if not self.coco_gt or not self.coco_dt:
            raise ValueError("Data not loaded. Call load_data() first.")

        coco_eval = COCOeval(self.coco_gt, self.coco_dt, eval_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def run_evaluation(self):
        """
        A high-level function to run the whole evaluation process.
        """
        self.load_data()
        self.evaluate()
