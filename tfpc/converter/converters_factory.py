"""
This module define a converter factory
"""
import logging
from converter.converter_proba_to_pred import ConverterProbaToPred
from converter.convert_inverse_weight_to_pred import ConverterInverseWeightToPred
from converter.converter_proba_to_pred_instance_weight import (
    ConverterProbaToPredInstanceWeight,
)
from converter.converter_lm_to_pred import ConverterLMToPred
from converter.converter_L5_medium_and_long import ConverterL5MostLikelyMediumAndLong
from converter.converter_no_changes import ConverterNoChanges
from converter.converter_multilabel_proba import ConverterMultiLabelProba


class ConvertersFactory:
    """
    This class is the converter factory
    """

    def get_converter(self, converter_type):
        logging.debug("J'essaye de get un converter de type : %s", (converter_type))
        if converter_type == "proba_to_pred":
            return ConverterProbaToPred()
        elif converter_type == "proba_to_pred_weighted":
            return ConverterProbaToPredInstanceWeight()
        elif converter_type == "LM_to_pred":
            return ConverterLMToPred()
        elif converter_type == "L5_medium_and_long":
            return ConverterL5MostLikelyMediumAndLong()
        elif converter_type == "inverse_weight_to_pred":
            return ConverterInverseWeightToPred()
        elif converter_type == "no_changes":
            return ConverterNoChanges()
        elif converter_type == "multilabel_proba":
            return ConverterMultiLabelProba()
        else:
            raise RuntimeError("Converter type unknown.")
