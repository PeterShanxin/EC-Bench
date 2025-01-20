"""
This module allow to compare multiple xp at once
"""
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.training_analyser import TrainingAnalyser
from utils.utils import save_fig
import numpy as np


class ComparingAnalyser:
    """
    This class compare multiple xp
    """

    def __init__(self, list_of_xp_path, comparaison_name, load_train=False):
        self.comparaison_name = comparaison_name
        self.list_of_analyser = []
        for path in list_of_xp_path:
            self.list_of_analyser.append(TrainingAnalyser(path, load_train=load_train))

        self.all_palettes_names = [
            "BrBG",
            "BuGn",
            "BuPu",
            "CMRmap",
            "Dark2",
            "GnBu",
            "GnBu_r",
            "Greens",
            "Greens_r",
            "Greys",
            "Greys_r",
            "OrRd",
            "OrRd_r",
            "Oranges",
            "Oranges_r",
            "PRGn",
            "PRGn_r",
            "Paired",
            "Paired_r",
            "Pastel1",
            "Pastel1_r",
            "Pastel2",
            "Pastel2_r",
            "PiYG",
            "PiYG_r",
            "PuBu",
            "PuBuGn",
            "PuBuGn_r",
            "PuBu_r",
            "PuOr",
            "PuOr_r",
            "PuRd",
            "PuRd_r",
            "Purples",
            "Purples_r",
            "RdBu",
            "RdBu_r",
            "RdGy",
            "RdGy_r",
            "RdPu",
            "RdPu_r",
            "RdYlBu",
            "RdYlBu_r",
            "RdYlGn",
            "RdYlGn_r",
            "Reds",
            "Reds_r",
            "Set1",
            "Set1_r",
            "Set2",
            "Set2_r",
            "Set3",
            "Set3_r",
            "Spectral",
            "Spectral_r",
            "Wistia",
            "Wistia_r",
            "YlGn",
            "YlGnBu",
            "YlGnBu_r",
            "YlGn_r",
            "YlOrBr",
            "YlOrBr_r",
            "YlOrRd",
            "YlOrRd_r",
            "afmhot",
            "afmhot_r",
            "autumn",
            "autumn_r",
            "binary",
            "binary_r",
            "bone",
            "bone_r",
            "brg",
            "brg_r",
            "bwr",
            "bwr_r",
            "cividis",
            "cividis_r",
            "cool",
            "cool_r",
            "coolwarm",
            "coolwarm_r",
            "copper",
            "copper_r",
            "crest",
            "crest_r",
            "cubehelix",
            "cubehelix_r",
            "flag",
            "flag_r",
            "flare",
            "flare_r",
            "gist_earth",
            "gist_earth_r",
            "gist_gray",
            "gist_gray_r",
            "gist_heat",
            "gist_heat_r",
            "gist_ncar",
            "gist_ncar_r",
            "gist_rainbow",
            "gist_rainbow_r",
            "gist_stern",
            "gist_stern_r",
            "gist_yarg",
            "gist_yarg_r",
            "gnuplot",
            "gnuplot2",
            "gnuplot2_r",
            "gnuplot_r",
            "gray",
            "gray_r",
            "hot",
            "hot_r",
            "hsv",
            "hsv_r",
            "icefire",
            "icefire_r",
            "inferno",
            "inferno_r",
            "jet",
            "jet_r",
            "magma",
            "magma_r",
            "mako",
            "mako_r",
            "nipy_spectral",
            "nipy_spectral_r",
            "ocean",
            "ocean_r",
            "pink",
            "pink_r",
            "plasma",
            "plasma_r",
            "prism",
            "prism_r",
            "rainbow",
            "rainbow_r",
            "rocket",
            "rocket_r",
            "seismic",
            "seismic_r",
            "spring",
            "spring_r",
            "summer",
            "summer_r",
            "tab10",
            "tab10_r",
            "tab20",
            "tab20_r",
            "tab20b",
            "tab20b_r",
            "tab20c",
            "tab20c_r",
            "terrain",
            "terrain_r",
            "turbo",
            "turbo_r",
            "twilight",
            "twilight_r",
            "twilight_shifted",
            "twilight_shifted_r",
            "viridis",
            "viridis_r",
            "vlag",
            "vlag_r",
            "winter",
            "winter_r",
        ]

    def plot_first_metric_on_all(self, only_show_valid=False):
        plt.title(self.comparaison_name + " : first metric")

        # Name the axes
        plt.ylabel("Metric value")
        plt.xlabel("Number of batch")

        for ind, analyser in enumerate(self.list_of_analyser):
            list_tasks = analyser.list_all_tasks()
            if len(list_tasks) > 1:
                for indi, task in enumerate(list_tasks):
                    print(indi, ":", task)
                indice_task = int(input("Choose an indice of a task(q to quit) : "))
            else:
                indice_task = 0

            # Trace loss
            df_all_loss = analyser.get_df_metric(indice_task, analyser.xp_name + "_")
            if only_show_valid:
                mask = [
                    el.split("_")[-2] == "dev" for el in list(df_all_loss["source"])
                ]
                df_all_loss = df_all_loss[mask]
                nb_curve = 1
            else:
                nb_curve = 2

            sns.lineplot(
                x="nb_batch",
                y="metric",
                hue="source",
                palette=sns.color_palette(self.all_palettes_names[ind + 1], nb_curve),
                data=df_all_loss,
                legend=True,
            )

        save_fig(
            "comparing_metric",
            "data/saved_figures/" + self.comparaison_name + "/",
        )
        plt.show()

    def plot_metric_for_ensemble(self):
        plt.title(self.comparaison_name + " : first metric")

        # Name the axes
        plt.ylabel("Metric value")
        plt.xlabel("Number of batch")

        group_one = []
        group_two = []

        for ind, analyser in enumerate(self.list_of_analyser):
            list_tasks = analyser.list_all_tasks()
            if len(list_tasks) > 1:
                for indi, task in enumerate(list_tasks):
                    print(indi, ":", task)
                indice_task = int(input("Choose an indice of a task(q to quit) : "))
            else:
                indice_task = 0

            # Trace loss
            df_all_loss = analyser.get_df_metric(indice_task, analyser.xp_name + "_")
            mask = [el.split("_")[-2] == "dev" for el in list(df_all_loss["source"])]
            df_all_loss = df_all_loss[mask]

            if "layer_norm" in analyser.xp_name:
                group_one.append(list(df_all_loss["metric"]))
            else:
                group_two.append(list(df_all_loss["metric"]))

        x_data = list(df_all_loss["nb_batch"])
        group_one = np.array(group_one)
        group_two = np.array(group_two)
        moyenne_g1 = group_one.mean(axis=0)
        std_g1 = group_one.std(axis=0)
        moyenne_g2 = group_two.mean(axis=0)
        std_g2 = group_two.std(axis=0)
        print(std_g1)

        plt.plot(x_data, moyenne_g1, label="layer_nrom", color="blue")
        plt.fill_between(
            x_data,
            (moyenne_g1 - 2 * std_g1),
            (moyenne_g1 + 2 * std_g1),
            color="blue",
            alpha=0.1,
        )

        plt.plot(x_data, moyenne_g2, label="classic", color="red")
        plt.fill_between(
            x_data,
            (moyenne_g2 - 2 * std_g2),
            (moyenne_g2 + 2 * std_g2),
            color="red",
            alpha=0.1,
        )
        plt.legend()
        plt.show()

    def plot_for_one_ensemble(self, metric):
        if metric:
            plt.title("Accuracy over the course of training")
        else:
            plt.title("Loss over the course of training")

        # Name the axes
        if metric:
            plt.ylabel("Accuracy")
        else:
            plt.ylabel("Cross entropy loss")
        plt.xlabel("Number of batch")

        group_dev = []
        group_train = []

        for ind, analyser in enumerate(self.list_of_analyser):
            list_tasks = analyser.list_all_tasks()
            if len(list_tasks) > 1:
                for indi, task in enumerate(list_tasks):
                    print(indi, ":", task)
                indice_task = int(input("Choose an indice of a task(q to quit) : "))
            else:
                indice_task = 0

            # Trace metric or loss
            if metric:
                df_all_loss = analyser.get_df_metric(
                    indice_task, analyser.xp_name + "_"
                )
                mask = [
                    el.split("_")[-2] == "train" for el in list(df_all_loss["source"])
                ]
                df_all_loss = df_all_loss[mask]
                group_dev.append(list(df_all_loss["metric"]))
            else:
                df_all_loss = analyser.get_df_loss(indice_task)
                mask = [el == "dev" for el in list(df_all_loss["source"])]
                df_all_loss = df_all_loss[mask]
                group_dev.append(list(df_all_loss["loss"]))

            x_data_dev = list(df_all_loss["nb_batch"])

            if metric:
                df_all_loss = analyser.get_df_metric(
                    indice_task, analyser.xp_name + "_"
                )
                mask = [
                    el.split("_")[-2] == "train" for el in list(df_all_loss["source"])
                ]
                df_all_loss = df_all_loss[mask]
                group_train.append(list(df_all_loss["metric"]))
            else:
                df_all_loss = analyser.get_df_loss(indice_task)
                mask = [el == "train" for el in list(df_all_loss["source"])]
                df_all_loss = df_all_loss[mask]
                group_train.append(list(df_all_loss["loss"]))

        print(len(group_train))
        print(len(group_dev))
        print(len(group_train[0]))
        print(len(group_dev[0]))

        x_data_train = list(df_all_loss["nb_batch"])
        group_train = np.array(group_train)
        group_dev = np.array(group_dev)
        moyenne_dev = group_dev.mean(axis=0)
        std_dev = group_dev.std(axis=0)
        moyenne_train = group_train.mean(axis=0)
        std_train = group_train.std(axis=0)

        plt.plot(x_data_dev, moyenne_dev, label="Dev", color="blue")
        plt.fill_between(
            x_data_dev,
            (moyenne_dev - 2 * std_dev),
            (moyenne_dev + 2 * std_dev),
            color="blue",
            alpha=0.1,
        )

        plt.plot(x_data_train, moyenne_train, label="Train", color="red")
        plt.fill_between(
            x_data_train,
            (moyenne_train - 2 * std_train),
            (moyenne_train + 2 * std_train),
            color="red",
            alpha=0.1,
        )
        plt.legend()
        plt.show()
