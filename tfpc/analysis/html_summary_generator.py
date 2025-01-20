"""
This module allow us to have a clear summary of finished experiences
"""
import os
import json
import logging
from os import listdir
from os.path import isfile, join
import torch
from analysis.training_analyser import TrainingAnalyser


class HtmlSummaryGenerator:
    """
    This class allow us to create html page to sum up experiences and their conclusion
    and also to have a dashboard on the advancement of experiences
    """

    def __init__(self):
        self.path_xp = "data/models/"
        self.path_save_html = "data/"
        self.name_saving = "/Train_logger_and_saver.pkl"
        self.meta_experiences_folder = "data/meta_experiences/"
        self.generate_dashboard_page()
        self.generate_meta_experiences_page()

    def generate_meta_experiences_page(self):
        experiments_descriptions = open(self.path_save_html + "experiments.html", "w")
        self.write_begin_experiences(experiments_descriptions)
        onlyfiles = [
            f
            for f in listdir(self.meta_experiences_folder)
            if isfile(join(self.meta_experiences_folder, f))
        ]
        for name_meta_xp in onlyfiles:
            fichier = open(self.meta_experiences_folder + name_meta_xp)
            logging.debug(
                "Path file to load is %s", self.meta_experiences_folder + name_meta_xp
            )
            config_meta = json.load(fichier)
            content = self.content_one_meta_xp(config_meta, name_meta_xp)
            experiments_descriptions.write(content)

        self.write_ending_experiences(experiments_descriptions)
        experiments_descriptions.close()

    def generate_dashboard_page(self):
        dashboard = open(self.path_save_html + "dashboard.html", "w")
        self.write_begining_dashboard(dashboard)
        list_directory = [x[0] for x in os.walk(self.path_xp + "fine_tune_models/")]
        list_directory = [x.split("/")[-1] for x in list_directory]
        for direc in list_directory:
            if direc[:4] != "test" and direc != "" and direc[0] != ".":
                state = self.get_state(direc)
                dashboard.write(state)
        self.write_ending_dashbaord(dashboard)
        dashboard.close()

    def get_state(self, directory):
        path_specific_xp = self.path_xp + "fine_tune_models/" + directory
        path_saving_file = path_specific_xp + self.name_saving
        print(path_saving_file)
        if not os.path.exists(path_saving_file):
            return (
                "<tr><td>"
                + directory
                + "</td><td>X</td><td></td><td></td><td></td><td>Unknown</td></tr>"
            )
        else:
            try:
                training_analyser = TrainingAnalyser(path_specific_xp)
                if training_analyser.finished_XP:  # pylint: disable=no-member
                    return (
                        "<tr><td>"
                        + directory
                        + "</td><td></td><td></td><td>"
                        + training_analyser.get_cluster()
                        + "</td><td>"
                        + self.create_td_metric_fine_tune(training_analyser)
                        + "</td></tr>"
                    )
                else:
                    return (
                        "<tr><td>"
                        + directory
                        + "</td><td></td><td>"
                        + training_analyser.get_cluster()
                        + "</td><td></td><td>"
                        + self.create_td_metric_fine_tune(training_analyser)
                        + "</td></tr>"
                    )
            except:
                return (
                    "<tr><td>"
                    + directory
                    + "</td><td></td><td></td><td></td><td>X</td><td>Unknown</td></tr>"
                )

    def write_begining_dashboard(self, dashboard):
        header = "<!DOCTYPE html><html lang='en'><head><style>td{text-align: center; }table, th, td {border: 1px solid black;}</style></head>"
        content = "<body><table><thead><tr><th>Experience name</th><th>Not launched</th><th>Running</th><th>Finished</th><th>Unknown</th><th>Metric on test</th></tr></thead><tbody>"
        dashboard.write(header + content)

    def write_ending_dashbaord(self, dashboard):
        content = "</tbody></table></body></html>"
        dashboard.write(content)

    def write_begin_experiences(self, html_experiment):
        header = "<!DOCTYPE html><html lang='en'><head><style>td{text-align: center; }table, th, td {border: 1px solid black;}</style></head><body>"
        html_experiment.write(header)

    def write_ending_experiences(self, html_experiment):
        content = "</body></html>"
        html_experiment.write(content)

    def content_one_meta_xp(self, config_meta, name_meta_xp):
        div_meta_xp = (
            "<div style='border:solid'><h2>" + name_meta_xp.split(".")[0] + "</h2>"
        )
        div_meta_xp += (
            "<p><p style='font-weight: bold;'>Description :</p>"
            + config_meta["description"]
            + "</p>"
        )
        div_meta_xp += "<p style='font-weight: bold;'>Result table :</p> "
        div_meta_xp += self.create_table_with_metric(config_meta["composition"])
        div_meta_xp += "<p style='font-weight: bold;'>Link figures :</p> "
        div_meta_xp += self.create_figure_meta_xp(config_meta["figures"])
        div_meta_xp += (
            "<p><p style='font-weight: bold;'>Conclusion :</p> "
            + config_meta["conclusion"]
            + "</p>"
        )
        div_meta_xp += "</div>"
        return div_meta_xp

    def create_table_with_metric(self, composition):
        div_meta_xp = "<table><thead><tr><th>Experience name</th><th>Metric on test</th><th>complete name</th></tr></thead><tbody>"
        for key, value in composition.items():
            simple_or_ensemble = value.split("/")[0]
            logging.debug("Model type is %s", simple_or_ensemble)
            div_meta_xp += "<tr><td>" + key + "</td>"
            path_specific_xp = self.path_xp + value
            if simple_or_ensemble == "fine_tune_models":
                training_analyser = TrainingAnalyser(path_specific_xp)
                div_meta_xp += self.create_td_metric_fine_tune(training_analyser)
                div_meta_xp += "</td><td>" + value + "</td></tr>"
            elif simple_or_ensemble == "ensembling_models":
                div_meta_xp += self.create_td_metric_ensembling(path_specific_xp)
                list_xp = torch.load(path_specific_xp + "/list_xp.pkl")
                div_meta_xp += (
                    "</td><td>" + value + "</br>" + str(list_xp) + "</td></tr>"
                )
            else:
                raise RuntimeError("Model not defined")
        div_meta_xp += "</table>"
        return div_meta_xp

    def create_figure_meta_xp(self, list_figures):
        html_code = ""
        for fig in list_figures:
            html_code += (
                "<iframe src='saved_figures/"
                + fig
                + "' width='50%' height='500px' ></iframe>"
            )
        return html_code

    def create_td_metric_fine_tune(self, training_analyser):
        dict_all_metric = training_analyser.get_all_test_metric()
        html = "<td>"
        for key_task, value_task in dict_all_metric.items():
            for key_metric, value_metric in value_task.items():
                html += (
                    key_task + "_" + key_metric + " : " + str(value_metric) + "</br>"
                )
        return html

    def create_td_metric_ensembling(self, path_specific_xp):
        dict_all_metric = torch.load(path_specific_xp + "/test_metrics.pkl")
        html = "<td>"
        for key_task, value_task in dict_all_metric.items():
            for key_metric, value_metric in value_task.items():
                html += (
                    key_task + "_" + key_metric + " : " + str(value_metric) + "</br>"
                )
        return html
