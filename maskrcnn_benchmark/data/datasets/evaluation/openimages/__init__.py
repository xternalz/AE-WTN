import logging

from .openimages_eval import do_openimages_evaluation


def openimages_evaluation(dataset, predictions, output_folder, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    return do_openimages_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
