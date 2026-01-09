import cv2
import gc
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from . import BaseModel
from app.core.registry import register_model


class _TaskStatus(Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _VideoSession:
    """Internal video session for SAM3."""

    def __init__(
        self,
        session_id: str,
        frames: List[np.ndarray],
        start_frame_index: int,
        predictor: Any,
        offload_video_to_cpu: bool = False,
    ):
        """Initialize video session.

        Args:
            session_id: Unique session identifier.
            frames: List of video frames as numpy arrays.
            start_frame_index: Starting frame index in original sequence.
            predictor: SAM3 video predictor instance.
        """
        self.session_id = session_id
        self.frames = frames
        self.start_frame_index = start_frame_index
        self.predictor = predictor
        self.offload_video_to_cpu = offload_video_to_cpu
        self.text_prompt: Optional[str] = None
        self.is_point_prompt: bool = False
        self.last_prompt_frame: Optional[int] = None
        self.prompt_frame_outputs: Optional[Dict[str, Any]] = None
        self.prompt_frame_params: Optional[Dict[str, Any]] = None
        self.rotation_cache: Dict[int, Dict[str, Any]] = {}
        self.low_conf_frames: Dict[int, int] = {}
        self.suppressed_obj_ids: set[int] = set()
        self.mask_cache: Dict[int, np.ndarray] = {}
        self.created_at = time.time()
        self.temp_dir: Optional[str] = None
        self._init_predictor_session()

    def _init_predictor_session(self):
        """Create temporary directory and initialize predictor session."""
        self.temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
        frame_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(frame_dir, f"{i:05d}.jpg")
            cv2.imwrite(frame_path, frame)

        self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frame_dir,
                session_id=self.session_id,
                offload_video_to_cpu=self.offload_video_to_cpu,
            )
        )
        logger.info(
            f"Created video session {self.session_id} with {len(self.frames)} frames"
        )

    def cleanup(self):
        """Clean up session resources."""
        try:
            self.predictor.handle_request(
                request=dict(type="close_session", session_id=self.session_id)
            )
        except Exception as e:
            logger.warning(f"Error closing session {self.session_id}: {e}")

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class _PropagationTask:
    """Internal propagation task for SAM3."""

    def __init__(
        self,
        task_id: str,
        session_id: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        """Initialize propagation task.

        Args:
            task_id: Unique task identifier.
            session_id: Video session identifier.
            start_frame: Optional start frame index.
            end_frame: Optional end frame index.
        """
        self.task_id = task_id
        self.session_id = session_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.status = _TaskStatus.PENDING
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.results: Dict[int, Any] = {}
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.lock = threading.Lock()
        self._cancelled = False

    def cancel(self):
        """Cancel the task."""
        with self.lock:
            self._cancelled = True
            if self.status == _TaskStatus.PROCESSING:
                self.status = _TaskStatus.CANCELLED

    def is_cancelled(self) -> bool:
        """Check if task is cancelled."""
        with self.lock:
            return self._cancelled


@register_model("segment_anything_3_video")
class SegmentAnything3Video(BaseModel):
    """Segment Anything Model 3 for video segmentation with text prompts."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize SAM3 video model.

        Args:
            config: Model configuration dictionary.
        """
        super().__init__(config)
        self.predictor = None
        self._sessions: Dict[str, _VideoSession] = {}
        self._tasks: Dict[str, _PropagationTask] = {}
        self._sessions_lock = threading.Lock()
        self._tasks_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._max_sessions = self._parse_int(
            self.params.get(
                "max_sessions", os.getenv("SAM3_MAX_SESSIONS", "1")
            ),
            1,
        )
        if self._max_sessions < 1:
            self._max_sessions = 1
        self._session_timeout = 1800
        self._propagate_chunk_size = self._parse_int(
            self.params.get(
                "propagate_chunk_size",
                os.getenv("SAM3_PROPAGATE_CHUNK_SIZE", "0"),
            ),
            0,
        )
        if self._propagate_chunk_size < 0:
            self._propagate_chunk_size = 0
        self._offload_video_to_cpu = self._parse_bool(
            self.params.get(
                "offload_video_to_cpu",
                os.getenv("SAM3_OFFLOAD_VIDEO_TO_CPU", "false"),
            )
        )
        self._async_loading_frames = self._parse_bool(
            self.params.get(
                "async_loading_frames",
                os.getenv("SAM3_ASYNC_LOADING_FRAMES", "true"),
            )
        )
        self._clear_cache_interval = self._parse_int(
            self.params.get(
                "clear_cache_interval",
                os.getenv("SAM3_CLEAR_CACHE_INTERVAL", "0"),
            ),
            0,
        )
        if self._clear_cache_interval < 0:
            self._clear_cache_interval = 0
        self._video_loader_type = str(
            self.params.get(
                "video_loader_type",
                os.getenv("SAM3_VIDEO_LOADER_TYPE", "cv2"),
            )
        ).strip().lower()
        if self._video_loader_type not in {"cv2", "torchcodec"}:
            logger.warning(
                f"Invalid SAM3 video_loader_type '{self._video_loader_type}', fallback to 'cv2'"
            )
            self._video_loader_type = "cv2"

    @staticmethod
    def _parse_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "y", "yes"}

    def load(self):
        """Load SAM3 video model and initialize components."""
        sam3_parent_dir = os.path.join(os.path.dirname(__file__))
        if sam3_parent_dir not in sys.path:
            sys.path.insert(0, sam3_parent_dir)

        from sam3.model_builder import build_sam3_video_predictor

        bpe_path = self.params.get("bpe_path")
        model_path = self.params.get("model_path")
        devices = self.params.get("devices", [0])

        if isinstance(devices, list) and devices:
            gpus_to_use = range(len(devices))
        elif isinstance(devices, int):
            gpus_to_use = range(devices)
        else:
            gpus_to_use = (
                range(torch.cuda.device_count())
                if torch.cuda.is_available()
                else range(1)
            )

        logger.info(
            f"Loading SAM3 video model from {model_path} on devices {gpus_to_use}"
        )
        apply_temporal_disambiguation = self._parse_bool(
            self.params.get("sam3_apply_temporal_disambiguation", True)
        )
        stability_thresh = self.params.get(
            "sam3_dynamic_multimask_stability_thresh", None
        )
        try:
            stability_thresh = (
                float(stability_thresh)
                if stability_thresh is not None
                else None
            )
        except (TypeError, ValueError):
            stability_thresh = None
        stability_delta = self.params.get(
            "sam3_dynamic_multimask_stability_delta", None
        )
        try:
            stability_delta = (
                float(stability_delta)
                if stability_delta is not None
                else None
            )
        except (TypeError, ValueError):
            stability_delta = None
        mf_threshold = self.params.get("sam3_mf_threshold", None)
        try:
            mf_threshold = (
                float(mf_threshold) if mf_threshold is not None else None
            )
        except (TypeError, ValueError):
            mf_threshold = None

        self.predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            bpe_path=bpe_path,
            checkpoint_path=model_path,
            async_loading_frames=self._async_loading_frames,
            video_loader_type=self._video_loader_type,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            dynamic_multimask_stability_thresh=stability_thresh,
            dynamic_multimask_stability_delta=stability_delta,
            mf_threshold=mf_threshold,
        )

        logger.info("SAM3 video model loaded successfully")

        if hasattr(self.predictor, "model") and hasattr(
            self.predictor.model, "model"
        ):
            model = self.predictor.model.model
            for name, param in model.named_parameters():
                if param.dtype != torch.float32 and "bias" in name:
                    param.data = param.data.float()

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute segmentation on single frame (not used for video).

        Args:
            image: Input image in BGR format.
            params: Inference parameters.

        Returns:
            Dictionary with empty shapes (video requires session-based API).
        """
        logger.warning(
            "Single frame predict called on video model. "
            "Use video API endpoints instead."
        )
        return {"shapes": [], "description": ""}

    def init_session(
        self, frames: List[np.ndarray], start_frame_index: int
    ) -> Dict[str, Any]:
        """Initialize a new video session.

        Args:
            frames: List of video frames as numpy arrays.
            start_frame_index: Starting frame index in original sequence.

        Returns:
            Dictionary with session_id, num_frames, start_frame_index.
        """
        with self._sessions_lock:
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_oldest_session()

            session_id = str(uuid.uuid4())

            if session_id in self._sessions:
                logger.warning(
                    f"Session {session_id} already exists, cleaning up"
                )
                self._sessions[session_id].cleanup()

            session = _VideoSession(
                session_id,
                frames,
                start_frame_index,
                self.predictor,
                offload_video_to_cpu=self._offload_video_to_cpu,
            )
            self._sessions[session_id] = session

        return {
            "session_id": session_id,
            "num_frames": len(frames),
            "start_frame_index": start_frame_index,
        }

    def add_prompt(
        self,
        session_id: str,
        text_prompt: str,
        frame_index: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add text prompt to a video frame.

        Args:
            session_id: Session identifier.
            text_prompt: Text prompt string.
            frame_index: Frame index to add prompt to.
            params: Additional parameters (conf_threshold, show_boxes, etc.).

        Returns:
            Dictionary with frame_index, masks list, and num_objects.
        """
        session = self._get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        if params.get("reset_tracker"):
            session.predictor.handle_request(
                request=dict(type="reset_session", session_id=session_id)
            )
            session.rotation_cache.clear()
            session.low_conf_frames.clear()
            session.suppressed_obj_ids.clear()
            session.mask_cache.clear()
            session.prompt_frame_outputs = None
            session.prompt_frame_params = None
            session.text_prompt = None

        relative_frame_index = frame_index - session.start_frame_index
        if relative_frame_index < 0 or relative_frame_index >= len(
            session.frames
        ):
            return {"error": f"Frame index {frame_index} out of range"}

        session.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )
        session.rotation_cache.clear()
        session.low_conf_frames.clear()
        session.suppressed_obj_ids.clear()
        session.mask_cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        response = session.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=relative_frame_index,
                text=text_prompt.rstrip("."),
            )
        )

        outputs = response.get("outputs", {})
        if not isinstance(outputs, dict):
            logger.error(f"Unexpected outputs format: {type(outputs)}")
            return {"error": f"Unexpected outputs format: {type(outputs)}"}

        out_binary_masks = outputs.get("out_binary_masks", [])
        out_probs = outputs.get("out_probs", [])
        out_obj_ids = self._normalize_obj_ids(outputs.get("out_obj_ids", []))
        out_boxes_xywh = outputs.get("out_boxes_xywh", [])
        out_tracker_probs = outputs.get("out_tracker_probs", None)

        if len(out_binary_masks) == 0:
            logger.warning("No masks returned from video prompt")
            return {"frame_index": frame_index, "masks": [], "num_objects": 0}

        session.text_prompt = text_prompt
        session.last_prompt_frame = frame_index
        session.prompt_frame_outputs = {
            "out_binary_masks": out_binary_masks,
            "out_probs": out_probs,
            "out_obj_ids": out_obj_ids,
            "out_boxes_xywh": out_boxes_xywh,
            "out_tracker_probs": out_tracker_probs,
        }
        session.prompt_frame_params = params.copy()

        inference_params = self._get_inference_params(params)
        orig_height, orig_width = self._get_frame_dimensions(session.frames)

        shapes = self._convert_outputs_to_shapes(
            out_binary_masks,
            out_probs,
            out_obj_ids,
            out_boxes_xywh,
            out_tracker_probs,
            text_prompt,
            inference_params["conf_threshold"],
            inference_params["tracker_conf_threshold"],
            inference_params["mask_min_area_ratio"],
            inference_params["mask_expand_ratio"],
            inference_params["mask_union_iou"],
            inference_params["mask_union_max_area_ratio"],
            inference_params["use_mask_bbox"],
            inference_params["show_boxes"],
            inference_params["show_masks"],
            inference_params["show_rotations"],
            inference_params["rotation_smooth"],
            inference_params["rotation_min_area"],
            inference_params["rotation_max_delta"],
            inference_params["rotation_lock_area_ratio"],
            inference_params["epsilon_factor"],
            orig_width,
            orig_height,
            session.suppressed_obj_ids,
            session.rotation_cache,
            session.mask_cache,
        )

        return {
            "frame_index": frame_index,
            "masks": shapes,
            "num_objects": len(shapes),
        }

    def add_point_prompt(
        self,
        session_id: str,
        points: List[List[float]],
        point_labels: List[int],
        obj_id: Optional[int],
        frame_index: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add point prompt to a video frame.

        Args:
            session_id: Session identifier.
            points: List of point coordinates in relative format (N, 2).
            point_labels: List of point labels (1 for positive, 0 for negative).
            obj_id: Object ID for the prompt.
            frame_index: Frame index to add prompt to.
            params: Additional parameters (conf_threshold, show_boxes, etc.).

        Returns:
            Dictionary with frame_index, masks list, and num_objects.
        """
        session = self._get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        relative_frame_index = frame_index - session.start_frame_index
        if relative_frame_index < 0 or relative_frame_index >= len(
            session.frames
        ):
            return {"error": f"Frame index {frame_index} out of range"}

        if not points or not point_labels:
            return {"error": "Points and point_labels are required"}

        if len(points) != len(point_labels):
            return {"error": "Points and point_labels must have same length"}

        if obj_id is None:
            obj_id = 99999

        if params.get("reset_tracker"):
            session.predictor.handle_request(
                request=dict(type="reset_session", session_id=session_id)
            )
            session.rotation_cache.clear()
            session.low_conf_frames.clear()
            session.suppressed_obj_ids.clear()
            session.mask_cache.clear()
            session.prompt_frame_outputs = None
            session.prompt_frame_params = None
            session.text_prompt = None

        session.low_conf_frames.clear()
        session.suppressed_obj_ids.clear()

        try:
            predictor_session = session.predictor._get_session(session_id)
            inference_state = predictor_session["state"]
            if (
                "cached_frame_outputs" not in inference_state
                or relative_frame_index
                not in inference_state["cached_frame_outputs"]
            ):
                if "cached_frame_outputs" not in inference_state:
                    inference_state["cached_frame_outputs"] = {}
                inference_state["cached_frame_outputs"][
                    relative_frame_index
                ] = {}
        except (AttributeError, RuntimeError, KeyError) as e:
            logger.warning(
                f"Could not initialize cache for frame {relative_frame_index}: {e}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        response = session.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=relative_frame_index,
                points=points,
                point_labels=point_labels,
                obj_id=obj_id,
            )
        )

        outputs = response.get("outputs", {})
        if not isinstance(outputs, dict):
            logger.error(f"Unexpected outputs format: {type(outputs)}")
            return {"error": f"Unexpected outputs format: {type(outputs)}"}

        out_binary_masks = outputs.get("out_binary_masks", [])
        out_probs = outputs.get("out_probs", [])
        out_obj_ids = self._normalize_obj_ids(outputs.get("out_obj_ids", []))
        out_boxes_xywh = outputs.get("out_boxes_xywh", [])
        out_tracker_probs = outputs.get("out_tracker_probs", None)

        if len(out_binary_masks) == 0:
            logger.warning("No masks returned from video point prompt")
            return {"frame_index": frame_index, "masks": [], "num_objects": 0}

        try:
            predictor_session = session.predictor._get_session(session_id)
            inference_state = predictor_session["state"]
            if "cached_frame_outputs" not in inference_state:
                inference_state["cached_frame_outputs"] = {}

            obj_id_to_mask = {}

            if len(out_obj_ids) > 0:
                import torch.nn.functional as F

                H_video = inference_state.get(
                    "orig_height", session.frames[0].shape[0]
                )
                W_video = inference_state.get(
                    "orig_width", session.frames[0].shape[1]
                )

                for i, obj_id in enumerate(out_obj_ids):
                    obj_id_int = (
                        int(obj_id) if hasattr(obj_id, '__int__') else obj_id
                    )
                    if i < len(out_binary_masks):
                        mask = out_binary_masks[i]
                        if isinstance(mask, np.ndarray):
                            mask_tensor = torch.from_numpy(mask).float()
                        elif isinstance(mask, torch.Tensor):
                            mask_tensor = mask.float()
                        else:
                            mask_tensor = torch.tensor(
                                mask, dtype=torch.float32
                            )

                        if mask_tensor.dim() == 2:
                            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                        elif mask_tensor.dim() == 3:
                            mask_tensor = mask_tensor.unsqueeze(0)

                        if mask_tensor.shape[-2:] != (H_video, W_video):
                            mask_tensor = F.interpolate(
                                mask_tensor,
                                size=(H_video, W_video),
                                mode="bilinear",
                                align_corners=False,
                            )

                        mask_bool = mask_tensor.squeeze() > 0.0
                        if isinstance(mask_bool, torch.Tensor):
                            mask_bool = mask_bool.to(torch.bool)
                        obj_id_to_mask[obj_id_int] = mask_bool

            frame_cache = inference_state["cached_frame_outputs"].setdefault(
                relative_frame_index, {}
            )
            frame_cache.update(obj_id_to_mask)
        except (AttributeError, RuntimeError, KeyError, Exception) as e:
            logger.warning(f"Could not initialize cache for all frames: {e}")

        session.is_point_prompt = True
        session.last_prompt_frame = frame_index
        session.prompt_frame_outputs = {
            "out_binary_masks": out_binary_masks,
            "out_probs": out_probs,
            "out_obj_ids": out_obj_ids,
            "out_boxes_xywh": out_boxes_xywh,
            "out_tracker_probs": out_tracker_probs,
        }
        session.prompt_frame_params = params.copy()

        inference_params = self._get_inference_params(params)
        orig_height, orig_width = self._get_frame_dimensions(session.frames)

        shapes = self._convert_outputs_to_shapes(
            out_binary_masks,
            out_probs,
            out_obj_ids,
            out_boxes_xywh,
            out_tracker_probs,
            "AUTOLABEL_OBJECT",
            inference_params["conf_threshold"],
            inference_params["tracker_conf_threshold"],
            inference_params["mask_min_area_ratio"],
            inference_params["mask_expand_ratio"],
            inference_params["mask_union_iou"],
            inference_params["mask_union_max_area_ratio"],
            inference_params["use_mask_bbox"],
            inference_params["show_boxes"],
            inference_params["show_masks"],
            inference_params["show_rotations"],
            inference_params["rotation_smooth"],
            inference_params["rotation_min_area"],
            inference_params["rotation_max_delta"],
            inference_params["rotation_lock_area_ratio"],
            inference_params["epsilon_factor"],
            orig_width,
            orig_height,
            session.suppressed_obj_ids,
            session.rotation_cache,
            session.mask_cache,
        )

        return {
            "frame_index": frame_index,
            "masks": shapes,
            "num_objects": len(shapes),
        }

    def start_propagation(
        self,
        session_id: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start video propagation task.

        Args:
            session_id: Session identifier.
            start_frame: Optional start frame index (absolute).
            end_frame: Optional end frame index (absolute).

        Returns:
            Dictionary with task_id and status.
        """
        session = self._get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        if not session.text_prompt and not session.is_point_prompt:
            return {"error": "No text prompt or point prompt set for session"}

        if start_frame is not None:
            start_frame = start_frame - session.start_frame_index
        if end_frame is not None:
            end_frame = end_frame - session.start_frame_index

        task_id = str(uuid.uuid4())
        task = _PropagationTask(task_id, session_id, start_frame, end_frame)

        with self._tasks_lock:
            self._tasks[task_id] = task

        logger.info(
            f"Submitting propagation task: task_id={task_id}, "
            f"session_id={session_id}"
        )

        try:
            future = self._executor.submit(
                self._run_propagation, task, session
            )

            def log_exception(fut):
                try:
                    fut.result()
                except Exception as e:
                    logger.opt(exception=True).error(
                        f"Propagation task {task_id} raised exception: {e}"
                    )

            future.add_done_callback(log_exception)
        except Exception as e:
            logger.error(f"Failed to submit propagation task {task_id}: {e}")
            with self._tasks_lock:
                task.status = _TaskStatus.FAILED
                task.error = f"Failed to submit task: {str(e)}"
            return {"error": str(e)}

        return {"task_id": task_id, "status": "processing"}

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get propagation task status.

        Args:
            task_id: Task identifier.

        Returns:
            Dictionary with status, progress, and results if completed.
        """
        with self._tasks_lock:
            task = self._tasks.get(task_id)

        if not task:
            return {"error": f"Task {task_id} not found"}

        session = self._get_session(task.session_id)
        start_frame_offset = session.start_frame_index if session else 0

        response_data = {
            "status": task.status.value,
            "progress": task.progress,
            "current_frame": (
                task.current_frame + start_frame_offset
                if task.current_frame
                else 0
            ),
            "total_frames": task.total_frames,
        }

        if task.status == _TaskStatus.COMPLETED:
            results = self._build_completed_results(task, session)
            response_data["results"] = results
        elif task.status == _TaskStatus.FAILED:
            response_data["error"] = task.error

        return response_data

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel propagation task.

        Args:
            task_id: Task identifier.

        Returns:
            Dictionary with success message or error.
        """
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task:
                task.cancel()
                return {"message": "Task cancelled"}
            return {"error": f"Task {task_id} not found"}

    def propagate_stream(
        self,
        session_id: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        """Stream video propagation results (generator for SSE).

        Args:
            session_id: Session identifier.
            start_frame: Optional start frame index (absolute).
            end_frame: Optional end frame index (absolute).

        Yields:
            Dictionary events with type, progress, and results.
        """
        session = self._get_session(session_id)
        if not session:
            yield {
                "type": "error",
                "message": f"Session {session_id} not found",
            }
            return

        if not session.text_prompt and not session.is_point_prompt:
            yield {
                "type": "error",
                "message": "No text prompt or point prompt set for session",
            }
            return

        rel_start = None
        rel_end = None
        if start_frame is not None:
            rel_start = start_frame - session.start_frame_index
        if end_frame is not None:
            rel_end = end_frame - session.start_frame_index

        total_frames = len(session.frames)
        start_frame_offset = session.start_frame_index
        orig_height, orig_width = self._get_frame_dimensions(session.frames)
        text_prompt = (
            session.text_prompt if session.text_prompt else "AUTOLABEL_OBJECT"
        )

        yield {
            "type": "started",
            "total_frames": total_frames,
            "start_frame_index": start_frame_offset,
        }

        bf16_context = None
        try:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(current_device)
                bf16_context = torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                )
                bf16_context.__enter__()

            prompt_frame_relative_idx = None
            prompt_absolute_idx = None
            if session.last_prompt_frame is not None:
                prompt_frame_relative_idx = (
                    session.last_prompt_frame - session.start_frame_index
                )
                prompt_absolute_idx = session.last_prompt_frame

            start_idx = (
                rel_start
                if rel_start is not None
                else (
                    prompt_frame_relative_idx
                    if prompt_frame_relative_idx is not None
                    else 0
                )
            )
            end_idx = (
                rel_end
                if rel_end is not None
                else max(total_frames - 1, 0)
            )
            if total_frames <= 0:
                yield {
                    "type": "error",
                    "message": "No frames available for propagation",
                }
                return
            start_idx = max(0, min(start_idx, total_frames - 1))
            end_idx = max(0, min(end_idx, total_frames - 1))
            if end_idx < start_idx:
                end_idx = start_idx

            chunk_size = self._propagate_chunk_size
            if session.prompt_frame_params:
                override = self._parse_int(
                    session.prompt_frame_params.get(
                        "propagate_chunk_size", 0
                    ),
                    0,
                )
                if override > 0:
                    chunk_size = override
            if chunk_size < 0:
                chunk_size = 0

            frame_count = 0
            results = {}

            chunk_start = start_idx
            while chunk_start <= end_idx:
                chunk_end = (
                    end_idx
                    if chunk_size <= 0
                    else min(chunk_start + chunk_size - 1, end_idx)
                )
                request_dict = dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_index=chunk_start,
                    max_frame_num_to_track=max(chunk_end - chunk_start, 0),
                )

                generator = session.predictor.handle_stream_request(
                    request=request_dict
                )

                for response in generator:
                    frame_idx = response.get("frame_index")
                    outputs = response.get("outputs")

                    if frame_idx is None or outputs is None:
                        continue

                    frame_count += 1
                    self._maybe_clear_cache(frame_count)
                    absolute_frame_idx = frame_idx + start_frame_offset

                    yield {
                        "type": "progress",
                        "current_frame": absolute_frame_idx,
                        "total_frames": total_frames,
                        "progress": frame_count / max(total_frames, 1),
                    }

                    if (
                        prompt_frame_relative_idx is not None
                        and frame_idx == prompt_frame_relative_idx
                        and session.prompt_frame_outputs is not None
                    ):
                        continue

                    out_binary_masks = outputs.get("out_binary_masks", [])
                    out_probs = outputs.get("out_probs", [])
                    out_obj_ids = self._normalize_obj_ids(
                        outputs.get("out_obj_ids", [])
                    )
                    out_boxes_xywh = outputs.get("out_boxes_xywh", [])
                    out_tracker_probs = outputs.get("out_tracker_probs", None)

                    params = self._get_inference_params(
                        session.prompt_frame_params
                    )
                    self._update_suppressed_obj_ids(
                        session,
                        out_binary_masks,
                        out_obj_ids,
                        out_tracker_probs,
                        params,
                        orig_width,
                        orig_height,
                    )
                    shapes = self._convert_outputs_to_shapes(
                        out_binary_masks,
                        out_probs,
                        out_obj_ids,
                        out_boxes_xywh,
                        out_tracker_probs,
                        text_prompt,
                        params["conf_threshold"],
                        params["tracker_conf_threshold"],
                        params["mask_min_area_ratio"],
                        params["mask_expand_ratio"],
                        params["mask_union_iou"],
                        params["mask_union_max_area_ratio"],
                        params["use_mask_bbox"],
                        params["show_boxes"],
                        params["show_masks"],
                        params["show_rotations"],
                        params["rotation_smooth"],
                        params["rotation_min_area"],
                        params["rotation_max_delta"],
                        params["rotation_lock_area_ratio"],
                        params["epsilon_factor"],
                        orig_width,
                        orig_height,
                        session.suppressed_obj_ids,
                        session.rotation_cache,
                        session.mask_cache,
                    )

                    results[absolute_frame_idx] = {"masks": shapes}

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()

                if chunk_end >= end_idx:
                    break
                chunk_start = chunk_end + 1

            if (
                session.prompt_frame_outputs is not None
                and prompt_absolute_idx is not None
            ):
                prompt_outputs = session.prompt_frame_outputs
                prompt_obj_ids = self._normalize_obj_ids(
                    prompt_outputs.get("out_obj_ids", [])
                )

                params = self._get_inference_params(
                    session.prompt_frame_params
                )
                prompt_shapes = self._convert_outputs_to_shapes(
                    prompt_outputs.get("out_binary_masks", []),
                    prompt_outputs.get("out_probs", []),
                    prompt_obj_ids,
                    prompt_outputs.get("out_boxes_xywh", []),
                    prompt_outputs.get("out_tracker_probs", None),
                    text_prompt,
                    params["conf_threshold"],
                    params["tracker_conf_threshold"],
                    params["mask_min_area_ratio"],
                    params["mask_expand_ratio"],
                    params["mask_union_iou"],
                    params["mask_union_max_area_ratio"],
                    params["use_mask_bbox"],
                    params["show_boxes"],
                    params["show_masks"],
                    params["show_rotations"],
                    params["rotation_smooth"],
                    params["rotation_min_area"],
                    params["rotation_max_delta"],
                    params["rotation_lock_area_ratio"],
                    params["epsilon_factor"],
                    orig_width,
                    orig_height,
                    session.suppressed_obj_ids,
                    session.rotation_cache,
                    session.mask_cache,
                )
                results[prompt_absolute_idx] = {"masks": prompt_shapes}

            yield {
                "type": "completed",
                "results": results,
            }

        except Exception as e:
            logger.opt(exception=True).error(
                f"Stream propagation error: {e}"
            )
            yield {"type": "error", "message": str(e)}
        finally:
            if bf16_context is not None:
                try:
                    bf16_context.__exit__(None, None, None)
                except Exception:
                    pass

    def cleanup_session(self, session_id: str) -> bool:
        """Clean up and remove a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if session was removed, False if not found.
        """
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.cleanup()
                return True
            return False

    def cleanup_all(self) -> Dict[str, Any]:
        """清理所有会话与任务，释放显存/内存缓存。"""
        cancelled_tasks = 0
        cleared_sessions = 0

        with self._tasks_lock:
            for task in self._tasks.values():
                task.cancel()
                cancelled_tasks += 1
            self._tasks.clear()

        with self._sessions_lock:
            for session in list(self._sessions.values()):
                try:
                    session.cleanup()
                except Exception as e:
                    logger.warning(f"Session cleanup error: {e}")
                cleared_sessions += 1
            self._sessions.clear()

        if hasattr(self, "predictor") and self.predictor:
            try:
                if hasattr(self.predictor, "_ALL_INFERENCE_STATES"):
                    self.predictor._ALL_INFERENCE_STATES.clear()
            except Exception as e:
                logger.warning(f"Predictor cleanup error: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        return {
            "sessions_cleared": cleared_sessions,
            "tasks_cancelled": cancelled_tasks,
        }

    def unload(self):
        """Release model resources."""
        with self._sessions_lock:
            for session in list(self._sessions.values()):
                session.cleanup()
            self._sessions.clear()

        with self._tasks_lock:
            for task in self._tasks.values():
                task.cancel()
        self._executor.shutdown(wait=True)
        self._tasks.clear()

        if hasattr(self, "predictor") and self.predictor:
            self.predictor.shutdown()
            del self.predictor
        logger.info("SAM3 video model unloaded")

    def _get_session(self, session_id: str) -> Optional[_VideoSession]:
        """Get session by ID with timeout check.

        Args:
            session_id: Session identifier.

        Returns:
            VideoSession instance or None if not found/expired.
        """
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session:
                if time.time() - session.created_at > self._session_timeout:
                    logger.warning(f"Session {session_id} expired")
                    session.cleanup()
                    del self._sessions[session_id]
                    return None
            return session

    def _cleanup_oldest_session(self):
        """Remove oldest session to make room for new one."""
        if not self._sessions:
            return

        oldest_id = min(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid].created_at,
        )
        logger.info(f"Cleaning up oldest session {oldest_id} to make room")
        session = self._sessions.pop(oldest_id, None)
        if session:
            session.cleanup()

    def _run_propagation(self, task: _PropagationTask, session: _VideoSession):
        """Run propagation task in background thread.

        Args:
            task: Propagation task instance.
            session: Video session instance.
        """
        logger.info(
            f"[TASK START] Propagation task started: "
            f"task_id={task.task_id}, "
            f"thread_id={threading.current_thread().ident}"
        )

        try:
            with task.lock:
                if task._cancelled:
                    logger.info(
                        f"Task {task.task_id} was cancelled before starting"
                    )
                    return
                task.status = _TaskStatus.PROCESSING

            total_frames = len(session.frames)
            with task.lock:
                task.total_frames = total_frames

            logger.info(
                f"Starting propagation for task {task.task_id}, "
                f"total_frames={total_frames}"
            )

            outputs_per_frame = {}
            frame_count = 0

            request_dict = dict(
                type="propagate_in_video",
                session_id=task.session_id,
                propagation_direction="forward",
            )
            if task.start_frame is not None:
                request_dict["start_frame_index"] = task.start_frame
            if task.end_frame is not None:
                request_dict["max_frame_num_to_track"] = task.end_frame

            bf16_context = None
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(current_device)
                bf16_context = torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                )
                bf16_context.__enter__()

            generator = session.predictor.handle_stream_request(
                request=request_dict
            )

            for response in generator:
                with task.lock:
                    if task._cancelled:
                        task.status = _TaskStatus.CANCELLED
                        logger.info(f"Task {task.task_id} was cancelled")
                        if bf16_context is not None:
                            try:
                                bf16_context.__exit__(None, None, None)
                            except Exception:
                                pass
                        return

                frame_idx = response.get("frame_index")
                outputs = response.get("outputs")

                if frame_idx is None or outputs is None:
                    continue

                outputs_per_frame[frame_idx] = outputs

                frame_count += 1
                self._maybe_clear_cache(frame_count)
                with task.lock:
                    task.current_frame = frame_idx
                    task.progress = (
                        frame_count / total_frames if total_frames > 0 else 0.0
                    )

            logger.info(
                f"Propagation finished for task {task.task_id}, "
                f"collected {len(outputs_per_frame)} frame results"
            )

            with task.lock:
                task.results = outputs_per_frame
                task.status = _TaskStatus.COMPLETED
                task.progress = 1.0

        except Exception as e:
            logger.opt(exception=True).error(
                f"Propagation task {task.task_id} failed: {e}"
            )
            with task.lock:
                task.status = _TaskStatus.FAILED
                task.error = str(e)
        finally:
            if bf16_context is not None:
                try:
                    bf16_context.__exit__(None, None, None)
                except Exception:
                    pass

    def _build_completed_results(
        self, task: _PropagationTask, session: Optional[_VideoSession]
    ) -> Dict[int, Dict[str, Any]]:
        """Build results dictionary for completed task.

        Args:
            task: Completed propagation task.
            session: Video session instance.

        Returns:
            Dictionary mapping frame index to masks.
        """
        results = {}
        start_frame_offset = session.start_frame_index if session else 0
        orig_height, orig_width = self._get_frame_dimensions(
            session.frames if session else None
        )
        text_prompt = session.text_prompt if session else "AUTOLABEL_OBJECT"

        prompt_frame_relative_idx = None
        prompt_absolute_idx = None
        if session and session.last_prompt_frame is not None:
            prompt_frame_relative_idx = (
                session.last_prompt_frame - session.start_frame_index
            )
            prompt_absolute_idx = session.last_prompt_frame

        for frame_idx in sorted(task.results.keys()):
            outputs = task.results[frame_idx]
            absolute_frame_idx = frame_idx + start_frame_offset

            if (
                prompt_frame_relative_idx is not None
                and frame_idx == prompt_frame_relative_idx
                and session
                and session.prompt_frame_outputs is not None
            ):
                continue

            out_binary_masks = outputs.get("out_binary_masks", [])
            out_probs = outputs.get("out_probs", [])
            out_obj_ids = self._normalize_obj_ids(
                outputs.get("out_obj_ids", [])
            )
            out_boxes_xywh = outputs.get("out_boxes_xywh", [])
            out_tracker_probs = outputs.get("out_tracker_probs", None)

            params = self._get_inference_params(
                session.prompt_frame_params if session else None
            )
            self._update_suppressed_obj_ids(
                session,
                out_binary_masks,
                out_obj_ids,
                out_tracker_probs,
                params,
                orig_width,
                orig_height,
            )
            shapes = self._convert_outputs_to_shapes(
                out_binary_masks,
                out_probs,
                out_obj_ids,
                out_boxes_xywh,
                out_tracker_probs,
                text_prompt,
                params["conf_threshold"],
                params["tracker_conf_threshold"],
                params["mask_min_area_ratio"],
                params["mask_expand_ratio"],
                params["mask_union_iou"],
                params["mask_union_max_area_ratio"],
                params["use_mask_bbox"],
                params["show_boxes"],
                params["show_masks"],
                params["show_rotations"],
                params["rotation_smooth"],
                params["rotation_min_area"],
                params["rotation_max_delta"],
                params["rotation_lock_area_ratio"],
                params["epsilon_factor"],
                orig_width,
                orig_height,
                session.suppressed_obj_ids if session else None,
                session.rotation_cache if session else None,
                session.mask_cache if session else None,
            )

            results[absolute_frame_idx] = {"masks": shapes}

        if (
            session
            and session.prompt_frame_outputs is not None
            and prompt_absolute_idx is not None
        ):
            prompt_outputs = session.prompt_frame_outputs
            prompt_obj_ids = self._normalize_obj_ids(
                prompt_outputs.get("out_obj_ids", [])
            )

            params = self._get_inference_params(session.prompt_frame_params)
            prompt_shapes = self._convert_outputs_to_shapes(
                prompt_outputs.get("out_binary_masks", []),
                prompt_outputs.get("out_probs", []),
                prompt_obj_ids,
                prompt_outputs.get("out_boxes_xywh", []),
                prompt_outputs.get("out_tracker_probs", None),
                text_prompt,
                params["conf_threshold"],
                params["tracker_conf_threshold"],
                params["mask_min_area_ratio"],
                params["mask_expand_ratio"],
                params["mask_union_iou"],
                params["mask_union_max_area_ratio"],
                params["use_mask_bbox"],
                params["show_boxes"],
                params["show_masks"],
                params["show_rotations"],
                params["rotation_smooth"],
                params["rotation_min_area"],
                params["rotation_max_delta"],
                params["rotation_lock_area_ratio"],
                params["epsilon_factor"],
                orig_width,
                orig_height,
                session.suppressed_obj_ids if session else None,
                session.rotation_cache if session else None,
                session.mask_cache if session else None,
            )
            results[prompt_absolute_idx] = {"masks": prompt_shapes}

        return results

    def _get_inference_params(
        self, prompt_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract inference parameters from prompt params or defaults.

        Args:
            prompt_params: Optional prompt frame parameters dictionary.

        Returns:
            Dictionary with conf_threshold, show_boxes, show_masks, epsilon_factor.
        """
        params = prompt_params or {}
        output_mode = (
            str(
                params.get(
                    "output_mode", self.params.get("output_mode", "")
                )
            )
            .strip()
            .lower()
        )
        show_boxes = params.get(
            "show_boxes", self.params.get("show_boxes", True)
        )
        show_masks = params.get(
            "show_masks", self.params.get("show_masks", False)
        )
        show_rotations = params.get(
            "show_rotations", self.params.get("show_rotations", False)
        )
        if output_mode:
            if output_mode in {"polygon", "mask", "seg"}:
                show_boxes = False
                show_masks = True
                show_rotations = False
            elif output_mode in {"obb", "rotation"}:
                show_boxes = False
                show_masks = False
                show_rotations = True
            elif output_mode in {"hbb", "rectangle", "box"}:
                show_boxes = True
                show_masks = False
                show_rotations = False

        return {
            "conf_threshold": params.get(
                "conf_threshold", self.params.get("conf_threshold", 0.25)
            ),
            "tracker_conf_threshold": params.get(
                "tracker_conf_threshold",
                self.params.get("tracker_conf_threshold", 0.0),
            ),
            "tracker_drop_frames": params.get(
                "tracker_drop_frames",
                self.params.get("tracker_drop_frames", 0),
            ),
            "tracker_drop_threshold": params.get(
                "tracker_drop_threshold",
                self.params.get("tracker_drop_threshold", None),
            ),
            "mask_min_area_ratio": params.get(
                "mask_min_area_ratio",
                self.params.get("mask_min_area_ratio", 0.0),
            ),
            "mask_expand_ratio": params.get(
                "mask_expand_ratio",
                self.params.get("mask_expand_ratio", 0.0),
            ),
            "mask_union_iou": params.get(
                "mask_union_iou", self.params.get("mask_union_iou", 0.0)
            ),
            "mask_union_max_area_ratio": params.get(
                "mask_union_max_area_ratio",
                self.params.get("mask_union_max_area_ratio", 0.0),
            ),
            "use_mask_bbox": params.get(
                "use_mask_bbox", self.params.get("use_mask_bbox", True)
            ),
            "show_boxes": show_boxes,
            "show_masks": show_masks,
            "show_rotations": show_rotations,
            "rotation_smooth": params.get(
                "rotation_smooth", self.params.get("rotation_smooth", 0.6)
            ),
            "rotation_min_area": params.get(
                "rotation_min_area", self.params.get("rotation_min_area", 10.0)
            ),
            "rotation_max_delta": params.get(
                "rotation_max_delta",
                self.params.get("rotation_max_delta", 0.523599),
            ),
            "rotation_lock_area_ratio": params.get(
                "rotation_lock_area_ratio",
                self.params.get("rotation_lock_area_ratio", 1.2),
            ),
            "epsilon_factor": params.get(
                "epsilon_factor", self.params.get("epsilon_factor", 0.001)
            ),
        }

    def _get_frame_dimensions(
        self, frames: Optional[List[np.ndarray]]
    ) -> Tuple[int, int]:
        """Get frame dimensions from frames list.

        Args:
            frames: Optional list of video frames.

        Returns:
            Tuple of (height, width) or default (1080, 1920).
        """
        if frames and len(frames) > 0:
            return len(frames[0]), len(frames[0][0])
        return 1080, 1920

    def _maybe_clear_cache(self, frame_count: int) -> None:
        if self._clear_cache_interval <= 0:
            return
        if frame_count % self._clear_cache_interval != 0:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _normalize_obj_ids(self, obj_ids: Any) -> np.ndarray:
        """Normalize object IDs to numpy array.

        Args:
            obj_ids: Object IDs in various formats.

        Returns:
            Numpy array of object IDs.
        """
        if isinstance(obj_ids, np.ndarray):
            return obj_ids
        elif isinstance(obj_ids, (list, tuple)):
            return np.array(obj_ids) if obj_ids else np.array([])
        else:
            return np.array([])

    def _update_suppressed_obj_ids(
        self,
        session: Optional[_VideoSession],
        out_binary_masks: np.ndarray,
        out_obj_ids: np.ndarray,
        out_tracker_probs: Optional[np.ndarray],
        params: Dict[str, Any],
        orig_width: int,
        orig_height: int,
    ) -> None:
        if session is None:
            return

        try:
            drop_frames = int(params.get("tracker_drop_frames", 0))
        except (TypeError, ValueError):
            drop_frames = 0
        if drop_frames <= 0:
            return

        tracker_threshold = params.get(
            "tracker_drop_threshold", params.get("tracker_conf_threshold", 0.0)
        )
        if tracker_threshold is None:
            tracker_threshold = params.get("tracker_conf_threshold", 0.0)
        try:
            tracker_threshold = max(float(tracker_threshold), 0.0)
        except (TypeError, ValueError):
            tracker_threshold = 0.0

        mask_min_area_ratio = params.get("mask_min_area_ratio", 0.0)
        try:
            mask_min_area_ratio = max(float(mask_min_area_ratio), 0.0)
        except (TypeError, ValueError):
            mask_min_area_ratio = 0.0

        image_area = float(max(orig_width * orig_height, 1))
        out_obj_ids = self._normalize_obj_ids(out_obj_ids)

        current_obj_ids = set()
        num_objects = (
            len(out_binary_masks)
            if isinstance(out_binary_masks, (list, np.ndarray))
            else 0
        )
        for i in range(num_objects):
            obj_id = None
            if i < len(out_obj_ids):
                try:
                    obj_id = int(out_obj_ids[i])
                except (ValueError, TypeError):
                    obj_id = None
            if obj_id is None:
                continue

            current_obj_ids.add(obj_id)

            tracker_prob = None
            if out_tracker_probs is not None:
                try:
                    tracker_prob = float(out_tracker_probs[i])
                except (IndexError, TypeError, ValueError):
                    tracker_prob = None

            mask_area_ratio = None
            try:
                mask = out_binary_masks[i]
                if isinstance(mask, np.ndarray):
                    mask_np = mask.astype(np.float32)
                else:
                    mask_np = np.array(mask, dtype=np.float32)
                mask_area_ratio = float((mask_np > 0.5).sum()) / image_area
            except Exception:
                mask_area_ratio = None

            low_conf = False
            if tracker_prob is not None and tracker_prob < tracker_threshold:
                low_conf = True
            if (
                mask_area_ratio is not None
                and mask_min_area_ratio > 0
                and mask_area_ratio < mask_min_area_ratio
            ):
                low_conf = True

            if low_conf:
                session.low_conf_frames[obj_id] = (
                    session.low_conf_frames.get(obj_id, 0) + 1
                )
            else:
                session.low_conf_frames[obj_id] = 0
                session.suppressed_obj_ids.discard(obj_id)

            if session.low_conf_frames[obj_id] >= drop_frames:
                session.suppressed_obj_ids.add(obj_id)
                session.rotation_cache.pop(obj_id + 1, None)

        missing_obj_ids = set(session.low_conf_frames.keys()) - current_obj_ids
        for obj_id in missing_obj_ids:
            session.low_conf_frames[obj_id] = (
                session.low_conf_frames.get(obj_id, 0) + 1
            )
            if session.low_conf_frames[obj_id] >= drop_frames:
                session.suppressed_obj_ids.add(obj_id)
                session.rotation_cache.pop(obj_id + 1, None)

    def _convert_outputs_to_shapes(
        self,
        out_binary_masks: np.ndarray,
        out_probs: np.ndarray,
        out_obj_ids: np.ndarray,
        out_boxes_xywh: np.ndarray,
        out_tracker_probs: Optional[np.ndarray],
        text_prompt: str,
        conf_threshold: float,
        tracker_conf_threshold: float,
        mask_min_area_ratio: float,
        mask_expand_ratio: float,
        mask_union_iou: float,
        mask_union_max_area_ratio: float,
        use_mask_bbox: bool,
        show_boxes: bool,
        show_masks: bool,
        show_rotations: bool,
        rotation_smooth: float,
        rotation_min_area: float,
        rotation_max_delta: float,
        rotation_lock_area_ratio: float,
        epsilon_factor: float,
        orig_width: int,
        orig_height: int,
        suppressed_obj_ids: Optional[set[int]] = None,
        rotation_cache: Optional[Dict[int, Dict[str, Any]]] = None,
        mask_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert SAM3 video outputs to shape dictionaries.

        Args:
            out_binary_masks: Binary masks array (N, H, W).
            out_probs: Confidence scores array (N,).
            out_obj_ids: Object IDs array (N,).
            out_boxes_xywh: Boxes in normalized xywh format (N, 4).
            text_prompt: Text prompt string.
            conf_threshold: Confidence threshold.
            tracker_conf_threshold: Tracker confidence threshold.
            mask_min_area_ratio: Minimum mask area ratio.
            use_mask_bbox: Whether to use mask-derived HBB.
            show_boxes: Whether to return bounding boxes.
            show_masks: Whether to return masks as polygons.
            epsilon_factor: Factor for polygon approximation.
            orig_width: Original image width.
            orig_height: Original image height.
            suppressed_obj_ids: Object IDs suppressed due to low confidence.

        Returns:
            List of shape dictionaries.
        """
        shapes = []

        if isinstance(out_binary_masks, (list, tuple)):
            if len(out_binary_masks) == 0:
                return shapes
        elif isinstance(out_binary_masks, np.ndarray):
            if out_binary_masks.size == 0:
                return shapes

        out_obj_ids = self._normalize_obj_ids(out_obj_ids)

        num_objects = (
            len(out_binary_masks)
            if isinstance(out_binary_masks, (list, np.ndarray))
            else 0
        )
        if num_objects == 0:
            return shapes

        image_area = float(orig_width * orig_height)
        if image_area <= 0:
            image_area = 1.0
        tracker_conf_threshold = max(float(tracker_conf_threshold), 0.0)
        mask_min_area_ratio = max(float(mask_min_area_ratio), 0.0)
        try:
            mask_expand_ratio = max(float(mask_expand_ratio), 0.0)
        except (TypeError, ValueError):
            mask_expand_ratio = 0.0
        try:
            mask_union_iou = max(float(mask_union_iou), 0.0)
        except (TypeError, ValueError):
            mask_union_iou = 0.0
        try:
            mask_union_max_area_ratio = max(
                float(mask_union_max_area_ratio), 0.0
            )
        except (TypeError, ValueError):
            mask_union_max_area_ratio = 0.0

        for i in range(num_objects):
            try:
                prob_val = float(out_probs[i])
            except (IndexError, TypeError, ValueError):
                prob_val = 1.0

            tracker_prob = None
            if out_tracker_probs is not None:
                try:
                    tracker_prob = float(out_tracker_probs[i])
                except (IndexError, TypeError, ValueError):
                    tracker_prob = None

            if tracker_prob is not None:
                if tracker_conf_threshold > 0 and tracker_prob < tracker_conf_threshold:
                    continue
                prob_val = min(prob_val, tracker_prob)

            if prob_val < conf_threshold:
                continue

            if text_prompt:
                label = text_prompt
                score = prob_val
            else:
                label = "AUTOLABEL_OBJECT"
                score = None

            obj_id = None
            group_id = None
            if i < len(out_obj_ids):
                try:
                    obj_id = int(out_obj_ids[i])
                    group_id = obj_id + 1
                except (ValueError, TypeError, IndexError):
                    pass

            if (
                suppressed_obj_ids is not None
                and obj_id is not None
                and obj_id in suppressed_obj_ids
            ):
                continue

            mask = out_binary_masks[i]
            if isinstance(mask, np.ndarray):
                mask_np = mask.astype(np.float32)
            else:
                mask_np = np.array(mask, dtype=np.float32)
            mask_binary = mask_np > 0.5

            mask_binary_box = mask_binary
            if mask_expand_ratio > 0:
                mask_binary_box = self._expand_mask(
                    mask_binary_box, mask_expand_ratio
                )

            mask_binary_mask = mask_binary_box
            if (
                show_masks
                and mask_union_iou > 0
                and mask_cache is not None
                and obj_id is not None
            ):
                prev_mask = mask_cache.get(obj_id)
                if (
                    prev_mask is not None
                    and isinstance(prev_mask, np.ndarray)
                    and prev_mask.shape == mask_binary_box.shape
                ):
                    iou = self._mask_iou(mask_binary_box, prev_mask)
                    if iou >= mask_union_iou:
                        union = np.logical_or(mask_binary_box, prev_mask)
                        if mask_union_max_area_ratio > 0:
                            base_area = float(mask_binary_box.sum())
                            union_area = float(union.sum())
                            if (
                                base_area > 0
                                and union_area
                                <= base_area * mask_union_max_area_ratio
                            ):
                                mask_binary_mask = union
                        else:
                            mask_binary_mask = union

            if mask_min_area_ratio > 0:
                area_mask = (
                    mask_binary_mask
                    if show_masks
                    else mask_binary_box
                )
                mask_area_ratio = float(area_mask.sum()) / image_area
                if mask_area_ratio < mask_min_area_ratio:
                    continue

            if (
                mask_cache is not None
                and obj_id is not None
                and (mask_binary_mask if show_masks else mask_binary_box).any()
            ):
                mask_cache[obj_id] = (
                    mask_binary_mask if show_masks else mask_binary_box
                )

            mask_np_mask = mask_binary_mask.astype(np.float32)
            mask_np_box = mask_binary_box.astype(np.float32)

            if show_masks:
                points = self._mask_to_polygon(
                    mask_np_mask, epsilon_factor
                )
                if points:
                    shapes.append(
                        {
                            "label": label,
                            "shape_type": "polygon",
                            "points": points,
                            "score": score,
                            "group_id": group_id,
                        }
                    )

            if show_rotations:
                rotation_data = self._mask_to_rotation(
                    mask_np_box, rotation_min_area
                )
                if rotation_data:
                    points_src = rotation_data.pop("points", None)
                    curr_box = None
                    if points_src is not None:
                        curr_box = self._box_from_points_with_angle(
                            points_src, rotation_data["angle"]
                        )
                        if curr_box:
                            rotation_data = curr_box
                    rotation_key = group_id if group_id is not None else i
                    if rotation_cache is not None:
                        prev = rotation_cache.get(rotation_key)
                        if prev and points_src is not None:
                            angle = rotation_data["angle"]
                            if (
                                rotation_lock_area_ratio > 0
                                and curr_box is not None
                            ):
                                prev_box = self._box_from_points_with_angle(
                                    points_src, prev["angle"]
                                )
                                if prev_box:
                                    area_ratio = prev_box["area"] / max(
                                        curr_box["area"], 1e-6
                                    )
                                    if (
                                        area_ratio
                                        <= rotation_lock_area_ratio
                                    ):
                                        angle = prev["angle"]
                            angle = self._smooth_rotation_angle(
                                prev["angle"],
                                angle,
                                rotation_smooth,
                                rotation_max_delta,
                            )
                            stable_box = self._box_from_points_with_angle(
                                points_src, angle
                            )
                            if stable_box:
                                if (
                                    rotation_lock_area_ratio > 0
                                    and curr_box is not None
                                ):
                                    area_ratio = stable_box["area"] / max(
                                        curr_box["area"], 1e-6
                                    )
                                    if (
                                        area_ratio
                                        <= rotation_lock_area_ratio
                                    ):
                                        rotation_data = stable_box
                                        rotation_data["angle"] = angle
                                    else:
                                        rotation_data = curr_box
                                else:
                                    rotation_data = stable_box
                                    rotation_data["angle"] = angle
                        rotation_cache[rotation_key] = rotation_data

                    box = rotation_data["box"]
                    shapes.append(
                        {
                            "label": label,
                            "shape_type": "rotation",
                            "points": [
                                [float(x), float(y)] for x, y in box
                            ],
                            "score": score,
                            "group_id": group_id,
                            "direction": rotation_data["angle"],
                        }
                    )

            if show_boxes:
                box_points = None
                if use_mask_bbox:
                    box_points = self._mask_to_hbb(mask_binary_box)
                if box_points is None:
                    try:
                        box_xywh = out_boxes_xywh[i]
                    except (IndexError, TypeError):
                        box_xywh = None
                    if box_xywh is not None:
                        x_norm, y_norm, w_norm, h_norm = box_xywh
                        x_min = float(x_norm * orig_width)
                        y_min = float(y_norm * orig_height)
                        x_max = float((x_norm + w_norm) * orig_width)
                        y_max = float((y_norm + h_norm) * orig_height)
                        box_points = [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max],
                        ]

                if box_points:
                    shapes.append(
                        {
                            "label": label,
                            "shape_type": "rectangle",
                            "points": box_points,
                            "score": score,
                            "group_id": group_id,
                        }
                    )

        return shapes

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        if mask_a is None or mask_b is None:
            return 0.0
        inter = np.logical_and(mask_a, mask_b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def _expand_mask(
        self, mask: np.ndarray, expand_ratio: float
    ) -> np.ndarray:
        if mask is None:
            return mask
        if expand_ratio <= 0:
            return mask
        height, width = mask.shape[:2]
        base = min(height, width)
        if base <= 0:
            return mask
        radius = int(round(base * expand_ratio))
        if radius < 1:
            return mask
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        expanded = cv2.dilate(
            mask.astype(np.uint8), kernel, iterations=1
        )
        return expanded.astype(bool)

    def _mask_to_polygon(
        self, mask: np.ndarray, epsilon_factor: float = 0.001
    ) -> List[List[float]]:
        """Convert binary mask to polygon points.

        Args:
            mask: Binary mask array.
            epsilon_factor: Factor for polygon approximation epsilon.

        Returns:
            List of polygon points.
        """
        mask_uint8 = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        largest_contour = max(contours, key=cv2.contourArea)
        if epsilon_factor > 0:
            epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        else:
            approx = largest_contour

        points = []
        for point in approx:
            x, y = point[0]
            points.append([float(x), float(y)])

        if points and points[0] != points[-1]:
            points.append(points[0])

        return points

    def _mask_to_hbb(self, mask: np.ndarray) -> Optional[List[List[float]]]:
        mask_uint8 = (mask > 0.5).astype(np.uint8)
        ys, xs = np.where(mask_uint8 > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x_min = float(xs.min())
        y_min = float(ys.min())
        x_max = float(xs.max())
        y_max = float(ys.max())
        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]

    def _box_from_points_with_angle(
        self, points: np.ndarray, angle_rad: float
    ) -> Optional[Dict[str, Any]]:
        if points is None or len(points) == 0:
            return None

        def compute_box(theta: float) -> Optional[Dict[str, Any]]:
            c = math.cos(theta)
            s = math.sin(theta)
            rot = np.array([[c, s], [-s, c]], dtype=np.float32)
            pts_rot = points @ rot.T
            min_xy = pts_rot.min(axis=0)
            max_xy = pts_rot.max(axis=0)
            w = float(max_xy[0] - min_xy[0])
            h = float(max_xy[1] - min_xy[1])
            if w <= 0 or h <= 0:
                return None
            cx = float((min_xy[0] + max_xy[0]) / 2)
            cy = float((min_xy[1] + max_xy[1]) / 2)
            inv = np.array([[c, -s], [s, c]], dtype=np.float32)
            center = np.array([cx, cy], dtype=np.float32) @ inv.T
            corners = np.array(
                [
                    [min_xy[0], min_xy[1]],
                    [max_xy[0], min_xy[1]],
                    [max_xy[0], max_xy[1]],
                    [min_xy[0], max_xy[1]],
                ],
                dtype=np.float32,
            )
            corners = corners @ inv.T
            return {
                "center": (float(center[0]), float(center[1])),
                "size": (w, h),
                "angle": float(theta % math.pi),
                "box": corners,
                "area": float(w * h),
            }

        box = compute_box(angle_rad)
        if box is None:
            return None
        if box["size"][0] < box["size"][1]:
            angle_rad = (angle_rad + math.pi / 2) % math.pi
            box = compute_box(angle_rad)
        return box

    def _smooth_rotation_angle(
        self, prev_angle: float, curr_angle: float, alpha: float, max_delta: float
    ) -> float:
        pa = float(prev_angle) % math.pi
        ca = float(curr_angle) % math.pi

        max_delta_val = float(max_delta) if max_delta is not None else 0.0
        if max_delta_val > math.pi:
            max_delta_val = math.radians(max_delta_val)
        if max_delta_val > 0:
            max_delta_val = min(max_delta_val, math.pi / 2)
            delta = ((ca - pa + math.pi / 2) % math.pi) - math.pi / 2
            if abs(delta) > max_delta_val:
                ca = (pa + math.copysign(max_delta_val, delta)) % math.pi

        if alpha <= 0 or alpha >= 1:
            angle = ca
        else:
            sin_val = (1 - alpha) * math.sin(2 * pa) + alpha * math.sin(
                2 * ca
            )
            cos_val = (1 - alpha) * math.cos(2 * pa) + alpha * math.cos(
                2 * ca
            )
            angle = 0.5 * math.atan2(sin_val, cos_val)
            if angle < 0:
                angle += math.pi
        return float(angle)

    def _mask_to_rotation(
        self, mask: np.ndarray, min_area: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        mask_uint8 = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_contour))
        if min_area > 0 and area < min_area:
            return None

        rect = cv2.minAreaRect(largest_contour)
        (_, _), (w, h), angle_deg = rect
        if w <= 0 or h <= 0:
            return None

        angle_rad = math.radians(angle_deg) % math.pi
        points_src = largest_contour.reshape(-1, 2).astype(np.float32)
        rotation_data = self._box_from_points_with_angle(
            points_src, angle_rad
        )
        if not rotation_data:
            return None
        rotation_data["points"] = points_src
        return rotation_data

    def _smooth_rotation(
        self,
        prev: Dict[str, Any],
        curr: Dict[str, Any],
        alpha: float,
        max_delta: float,
    ) -> Dict[str, Any]:
        px, py = prev["center"]
        cx, cy = curr["center"]
        pw, ph = prev["size"]
        cw, ch = curr["size"]
        pa = float(prev["angle"]) % math.pi
        ca = float(curr["angle"]) % math.pi

        max_delta_val = float(max_delta) if max_delta is not None else 0.0
        if max_delta_val > math.pi:
            max_delta_val = math.radians(max_delta_val)
        if max_delta_val > 0:
            max_delta_val = min(max_delta_val, math.pi / 2)
            delta = ((ca - pa + math.pi / 2) % math.pi) - math.pi / 2
            if abs(delta) > max_delta_val:
                ca = (pa + math.copysign(max_delta_val, delta)) % math.pi

        if alpha <= 0 or alpha >= 1:
            angle = ca
            nx, ny = cx, cy
            nw, nh = cw, ch
        else:
            sin_val = (1 - alpha) * math.sin(2 * pa) + alpha * math.sin(
                2 * ca
            )
            cos_val = (1 - alpha) * math.cos(2 * pa) + alpha * math.cos(
                2 * ca
            )
            angle = 0.5 * math.atan2(sin_val, cos_val)
            if angle < 0:
                angle += math.pi

            nx = (1 - alpha) * px + alpha * cx
            ny = (1 - alpha) * py + alpha * cy
            nw = (1 - alpha) * pw + alpha * cw
            nh = (1 - alpha) * ph + alpha * ch

        angle_deg = math.degrees(angle)
        box = cv2.boxPoints(((nx, ny), (nw, nh), angle_deg))
        return {
            "center": (float(nx), float(ny)),
            "size": (float(nw), float(nh)),
            "angle": float(angle),
            "box": box,
        }
