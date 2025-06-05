from functools import partial
from typing import Dict, TYPE_CHECKING, Any
import json
import numpy as np
import pandas as pd
import torch
from lightning.fabric import Fabric

from abc import abstractmethod, ABC

from tracklab.pipeline import Pipeline

if TYPE_CHECKING:
    from tracklab.callbacks import Callback

from tracklab.datastruct import TrackerState
def save_predictions_to_json(predictions_df, output_path):
    pred_dict = {}

    for image_id, row in predictions_df.iterrows():
        image_entry = {}

        # Ajoute les keypoints ("lines") s'ils existent
        if "lines" in row and isinstance(row["lines"], dict):
            image_entry["lines"] = row["lines"]

        # Ajoute les paramètres de caméra ("camera") s'ils existent
        if "parameters" in row and isinstance(row["parameters"], dict):
            image_entry["camera"] = row["parameters"]

        # Ajoute les bbox terrain ("bbox_pitch") si présentes
        if "bbox_pitch" in row and isinstance(row["bbox_pitch"], list):
            image_entry["bbox_pitch"] = row["bbox_pitch"]

        # Ajoute au dictionnaire final
        pred_dict[str(image_id)] = image_entry

    with open(output_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    print(f"✅ Prédictions sauvegardées dans : {output_path}")

def merge_dataframes(main_df, appended_piece):
    # Convert appended_piece to a DataFrame if it's not already
    if isinstance(appended_piece, pd.Series):
        appended_piece = pd.DataFrame(appended_piece).T
    elif isinstance(appended_piece, list):  # list of Series or DataFrames
        if len(appended_piece) > 0:
            appended_piece = pd.concat(
                [s.to_frame().T if type(s) is pd.Series else s for s in appended_piece]
            )
        else:
            appended_piece = pd.DataFrame()

    # Append the columns of the df
    new_columns = appended_piece.columns.difference(main_df.columns)
    main_df.loc[:, new_columns] = np.nan

    # Append the rows of the df
    new_index = set(appended_piece.index).difference(main_df.index)
    for index in new_index:
        main_df.loc[index] = np.nan

    # Update all the values (appended_piece overrides)
    main_df.update(appended_piece)
    return main_df


class TrackingEngine(ABC):
    """Manages the full tracking pipeline.

    After initializing the :class:`TrackingEngine`, users should call :class:`track_dataset`
    which will track each video in turn. The call stack looks like :

    track_dataset
    |
    video_step
    |
    -> detect_multi_step -> detect_single_step -> reid_step -> track_step

    Implementors of :class:`TrackingEngine` will need to *at least* implement
    :func:`video_loop`.
    for example, an online engine will simply call each step in turn for every image in
    a video. An offline engine might instead call each step for all the images before
    doing the next step in the pipeline.

    You should take care to implement the different callback hooks, by calling::

        self.fabric.call("a_callback_function", *args, **kwargs)

    Args:
        detect_multi_model: The bbox/pose detection model
        detect_single_model: The pose detection model
        reid_model: Reid model
        track_model: tracking model
        tracker_state: contains inputs and outputs
        callbacks: called at different steps
        num_workers: number of workers for preprocessing
    """

    def __init__(
        self,
        modules: Pipeline,
        tracker_state: TrackerState,
        num_workers: int,
        callbacks: "Dict[Callback]" = None,
    ):
        # super().__init__()
        self.module_names = [module.name for module in modules]
        self.callbacks = callbacks or {}
        callbacks_before = [c for c in callbacks.values() if not c.after_saved_state]
        callbacks_after = [c for c in callbacks.values() if c.after_saved_state]
        callbacks = callbacks_before + [tracker_state] + callbacks_after

        self.fabric = Fabric(callbacks=callbacks)
        self.callback = partial(self.fabric.call, engine=self)
        self.num_workers = num_workers
        self.tracker_state = tracker_state
        self.img_metadatas = tracker_state.image_metadatas
        self.video_metadatas = tracker_state.video_metadatas
        self.models = {model.name: model for model in modules}
        self.datapipes = {}
        self.dataloaders = {}
        self.all_image_preds = []
        for model_name, model in self.models.items():
            self.datapipes[model_name] = getattr(model, "datapipe", None)
            self.dataloaders[model_name] = getattr(model, "dataloader", lambda **kwargs: ...)(engine=self)

    def track_dataset(self):
        """Run tracking on complete dataset."""
        self.callback("on_dataset_track_start")
        for i, (video_idx, video_metadata) in enumerate(
            self.video_metadatas.iterrows()
        ):
            with self.tracker_state(video_idx) as tracker_state:
                self.callback(
                    "on_video_loop_start",
                    video_metadata=video_metadata,
                    video_idx=video_idx,
                    index=i,
                )
                detections, image_pred = self.video_loop(tracker_state, video_metadata, video_idx)


                self.callback(
                    "on_video_loop_end",
                    video_metadata=video_metadata,
                    video_idx=video_idx,
                    detections=detections,
                    image_pred=image_pred,
                )
                if "lines" in image_pred.columns:
                    self.all_image_preds.append(image_pred)

        self.callback("on_dataset_track_end")
        all_preds_df = pd.concat(self.all_image_preds)
        save_predictions_to_json(all_preds_df, "calib_pred.json")


    @abstractmethod
    def video_loop(
        self, tracker_state: TrackerState, video_metadata: pd.Series, video_id: int
    ) -> pd.DataFrame:
        """Run tracking on one video.

        The pipeline for each video looks like :

        detect_multi -> (detect_single) -> reid -> track

        Args:
            tracker_state (TrackerState): tracker state object
            video_metadata (pd.Series): metadata for the video
            video_id (int): id of the video

        Returns:
            detections: a dataframe of all detections
        """
        pass

    def default_step(self, batch: Any, task: str, detections: pd.DataFrame,
                     image_pred: pd.DataFrame, **kwargs):
        model = self.models[task]
        self.callback(f"on_module_step_start", task=task, batch=batch)

        idxs, batch = batch
        idxs = idxs.cpu() if isinstance(idxs, torch.Tensor) else idxs
        #print("Ehhhh c'est bien ici qu'on choisi la task et pou rle moment c'est ",task)
        if model.level == "image":
            batch_metadatas = image_pred.loc[list(idxs)]  # self.img_metadatas.loc[idxs]
            if len(detections) > 0:
                batch_input_detections = detections.loc[
                    np.isin(detections.image_id, batch_metadatas.index)
                ]
            else:
                batch_input_detections = detections
            batch_detections = self.models[task].process(
                batch,
                batch_input_detections,
                batch_metadatas)
            #print(batch_detections)
        else:

            batch_detections = detections.loc[list(idxs)]
            if not image_pred.empty:
                batch_metadatas = image_pred.loc[np.isin(image_pred.index, batch_detections.image_id)]
            else:
                batch_metadatas = image_pred
            batch_detections = self.models[task].process(
                batch=batch,
                detections=batch_detections,
                metadatas=batch_metadatas,
                **kwargs,
            )
        if isinstance(batch_detections, tuple):
            batch_detections, batch_metadatas = batch_detections
            image_pred = merge_dataframes(image_pred, batch_metadatas)
        detections = merge_dataframes(detections, batch_detections)
        self.callback(
            f"on_module_step_end", task=task, batch=batch, detections=detections
        )
        return detections, image_pred
