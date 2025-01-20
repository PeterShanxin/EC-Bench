import numpy as np
from sklearn.metrics import classification_report


class ECPredF1:
    def __init__(self):
        self.reset_metric()

    def reset_metric(self):
        self.TP = np.zeros(4)
        self.FN = np.zeros(4)
        self.TN = np.zeros(4)
        self.FP = np.zeros(4)
        self.correct = np.zeros(4)
        self.incorrect = np.zeros(4)

    def step(self, true_label, prediction):
        class_zero = "0.0.0.0"

        for lvl in range(1, 5):
            label_truncated = ".".join(true_label.split(".")[:lvl])
            pred_truncated = ".".join(prediction.split(".")[:lvl])
            class_zero_truncated = ".".join(class_zero.split(".")[:lvl])
            if pred_truncated == label_truncated:
                self.correct[lvl - 1] += 1
            else:
                self.incorrect[lvl - 1] += 1
            if (
                label_truncated == class_zero_truncated
                and pred_truncated == class_zero_truncated
            ):
                self.TN[lvl - 1] += 1
            elif (
                label_truncated == class_zero_truncated
                and pred_truncated != class_zero_truncated
            ):
                self.FP[lvl - 1] += 1
            elif label_truncated == pred_truncated:
                self.TP[lvl - 1] += 1
            else:
                self.FN[lvl - 1] += 1

    def get_main_metric(self):
        all_metrics = self.get_metric_at_lvl(4)
        return all_metrics["f1_score"]

    def get_all_metrics(self, level_max):
        for lvl in range(1, level_max + 1):
            all_metrics = self.get_metric_at_lvl(lvl)
            print("At level ", lvl, ":")
            print(all_metrics["macro avg"])

    def get_metric_at_lvl(self, lvl):
        recall = self.TP[lvl - 1] / (self.TP[lvl - 1] + self.FN[lvl - 1])
        precision = self.TP[lvl - 1] / (self.TP[lvl - 1] + self.FP[lvl - 1])
        f1_score = 2 * ((precision * recall) / (precision + recall))
        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    def log_all_metrics(self, filepath):
        fichier_results_metrics = open(filepath, "w")

        fichier_results_metrics.write("level,precision,recall,f1\n")
        for lvl in range(1, 5):
            all_metrics = self.get_metric_at_lvl(lvl)
            fichier_results_metrics.write(
                str(lvl)
                + ","
                + str(all_metrics["precision"])
                + ","
                + str(all_metrics["recall"])
                + ","
                + str(all_metrics["f1_score"])
                + "\n"
            )
        fichier_results_metrics.close()


class MacroAvg:
    def __init__(self, separator="."):
        self.reset_metric()
        self.separator = separator

    def reset_metric(self):
        self.y_pred = []
        self.y_true = []

    def step(self, true_label, prediction):
        prediction = str(prediction)

        if len(prediction.split(".")) == 2:
            prediction = prediction + ".0.0"
        self.y_true.append(true_label)
        self.y_pred.append(prediction)

    def get_main_metric(self):
        label_truncated = (
            np.array(self.y_true)
            == "0" + self.separator + "0" + self.separator + "0" + self.separator + "0"
        )
        pred_truncated = (
            np.array(self.y_pred)
            == "0" + self.separator + "0" + self.separator + "0" + self.separator + "0"
        )
        rep = classification_report(
            label_truncated, pred_truncated, output_dict=True, zero_division=0.0  # 0
        )
        macro_f1 = rep["macro avg"]["f1-score"]

        return macro_f1

    def get_all_metrics(self, level_max):
        for lvl in range(1, level_max + 1):
            print("At level ", lvl, ":")
            all_metrics = self.get_metric_at_lvl(lvl)
            print("macro_avg :", all_metrics["macro avg"])
            # print("weighted_avg :", all_metrics["weighted avg"])
            print("accuracy :", all_metrics["accuracy"])
            # print("Number of different ec in test set :", len(all_metrics.keys()) - 3)

    def get_metric_at_lvl(self, lvl):
        if lvl == 0:
            label_truncated = (
                np.array(self.y_true)
                == "0"
                + self.separator
                + "0"
                + self.separator
                + "0"
                + self.separator
                + "0"
            )
            pred_truncated = (
                np.array(self.y_pred)
                == "0"
                + self.separator
                + "0"
                + self.separator
                + "0"
                + self.separator
                + "0"
            )
        else:
            label_truncated = [
                ".".join(true_label.split(self.separator)[:lvl])
                for true_label in self.y_true
            ]
            pred_truncated = [
                ".".join(prediction.split(self.separator)[:lvl])
                for prediction in self.y_pred
            ]

        rep = classification_report(
            label_truncated,
            pred_truncated,
            output_dict=True,
            zero_division=0.0,
        )

        return rep

    def log_all_metrics(self, filepath, which_level):
        if which_level == "0":
            all_levels = [0]
        elif which_level == "others":
            all_levels = list(range(1, 5))
        else:
            raise RuntimeError("Level unknown")
        fichier_results_metrics = open(filepath, "w")

        fichier_results_metrics.write(
            "level,metric_type,precision,recall,f1,accuracy\n"
        )
        for lvl in all_levels:
            all_metrics = self.get_metric_at_lvl(lvl)
            accuracy = all_metrics["accuracy"]
            for metric_type in all_metrics.keys():
                if isinstance(all_metrics[metric_type], dict):
                    fichier_results_metrics.write(
                        str(lvl)
                        + ","
                        + metric_type
                        + ","
                        + str(all_metrics[metric_type]["precision"])
                        + ","
                        + str(all_metrics[metric_type]["recall"])
                        + ","
                        + str(all_metrics[metric_type]["f1-score"])
                        + ","
                        + str(accuracy)
                        + "\n"
                    )
        fichier_results_metrics.close()
