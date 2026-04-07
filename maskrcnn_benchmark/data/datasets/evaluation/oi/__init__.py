from .oi_eval import do_oi_evaluation


def oi_evaluation(cfg, dataset, predictions, output_folder, logger, **kwargs):
    iou_types = kwargs.pop("iou_types", ("bbox",))
    return do_oi_evaluation(
        cfg=cfg,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types,
    )
