#!/usr/bin/env python3
"""Unified Move Mirror implementation using Mediapipe.

This script replaces the original multi-module Nuxt.js + PoseNet project with a
standalone Python utility.  It performs three primary jobs:

1. Estimate human poses for images with Mediapipe and maintain an on-disk pose
   database that mirrors the original posedata.json file.
2. Build and persist a VP-tree search structure that can be used to retrieve the
   closest poses to a query pose.
3. Offer helper utilities for managing the mirror image library, including
   expanding the database when new images are added and pruning images when they
   are no longer needed.

All major functions are documented and separated so they can be imported and
re-used by other tools or automation scripts.
"""

# ---------------------------------------------------------------------------
# Imports and global constants are defined first so every dependency is easy to
# audit.  The project now depends only on standard Python libraries plus
# Mediapipe, OpenCV, and NumPy.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.solutions import drawing_styles as mp_drawing_styles
from mediapipe.solutions import drawing_utils as mp_drawing


# ---------------------------------------------------------------------------
# File system layout mirrors the previous repository structure so existing
# assets continue to work without modification.  Data files are stored in a
# dedicated "data" directory created at runtime.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MIRROR_IMAGE_DIR = BASE_DIR / "assets" / "images" / "mirror-poses"
DEBUG_IMAGE_DIR = BASE_DIR / "assets" / "images" / "debug"
DATA_DIR = BASE_DIR / "data"
POSE_DATA_PATH = DATA_DIR / "posedata.json"
PREBUILT_TREE_PATH = DATA_DIR / "prebuilt_vptree.json"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for directory in (MIRROR_IMAGE_DIR, DEBUG_IMAGE_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PoseRecord represents a single normalized pose entry in the database.  The
# structure matches the legacy posedata.json format to keep compatibility with
# existing tooling and expectations.
# ---------------------------------------------------------------------------

@dataclass
class PoseRecord:
    image: str
    vector_xy: List[float]
    vector_confidence: List[float]

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "vectorXY": self.vector_xy,
            "vectorConfidence": self.vector_confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PoseRecord":
        return cls(
            image=payload["image"],
            vector_xy=list(payload["vectorXY"]),
            vector_confidence=list(payload["vectorConfidence"]),
        )


# ---------------------------------------------------------------------------
# Normalization helpers convert Mediapipe landmarks into translation and scale
# invariant vectors similar to what posenet-similarity produced previously.
# ---------------------------------------------------------------------------

def _landmark_confidence(landmark: landmark_pb2.NormalizedLandmark) -> float:
    visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
    presence = float(getattr(landmark, "presence", 1.0) or 0.0)
    visibility = max(0.0, min(visibility, 1.0))
    presence = max(0.0, min(presence, 1.0))
    return visibility * presence if presence != 0 else visibility


def normalize_landmarks(
    landmarks: Sequence[landmark_pb2.NormalizedLandmark],
) -> Tuple[List[float], List[float]]:
    """Convert Mediapipe landmarks into normalized vectors and confidences."""

    if not landmarks:
        return [], []

    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    confidences = np.array([_landmark_confidence(lm) for lm in landmarks], dtype=np.float32)

    center = np.mean(coords, axis=0)
    coords -= center

    norms = np.linalg.norm(coords, axis=1)
    scale = float(np.max(norms))
    if scale > 1e-6:
        coords /= scale

    vector_xy = coords.flatten().astype(np.float32).tolist()
    vector_confidence = confidences.astype(np.float32).tolist()
    return vector_xy, vector_confidence


# ---------------------------------------------------------------------------
# PoseEstimator wraps Mediapipe's Pose solution and exposes convenience methods
# for estimating poses from both image files and in-memory frames.
# ---------------------------------------------------------------------------


class PoseEstimator:
    def __init__(
        self,
        static_image_mode: bool = True,
        model_complexity: int = 2,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self._pose_kwargs = dict(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
        )
        self._pose: Optional[mp.solutions.pose.Pose] = None

    def __enter__(self) -> "PoseEstimator":
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_initialized(self) -> None:
        if self._pose is None:
            self._pose = mp.solutions.pose.Pose(**self._pose_kwargs)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def estimate_from_file(
        self,
        image_path: Path,
        *,
        return_landmarks: bool = False,
    ) -> Optional[Union[PoseRecord, Tuple[PoseRecord, landmark_pb2.NormalizedLandmarkList]]]:
        """Estimate a pose from the provided image file."""

        self._ensure_initialized()
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return self.estimate_from_array(
            image_bgr,
            image_path.name,
            return_landmarks=return_landmarks,
        )

    def estimate_from_array(
        self,
        image_bgr: np.ndarray,
        label: str = "frame",
        *,
        return_landmarks: bool = False,
    ) -> Optional[Union[PoseRecord, Tuple[PoseRecord, landmark_pb2.NormalizedLandmarkList]]]:
        """Estimate a pose from an in-memory BGR image."""

        self._ensure_initialized()
        if image_bgr is None or image_bgr.size == 0:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        vector_xy, vector_confidence = normalize_landmarks(results.pose_landmarks.landmark)
        record = PoseRecord(label, vector_xy, vector_confidence)
        if return_landmarks:
            return record, results.pose_landmarks
        return record


# ---------------------------------------------------------------------------
# PoseDatabase maintains the posedata.json file, offering CRUD-style helpers so
# other parts of the script do not interact with JSON serialization directly.
# ---------------------------------------------------------------------------


class PoseDatabase:
    def __init__(self, path: Path = POSE_DATA_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[PoseRecord] = self._load()

    def _load(self) -> List[PoseRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [PoseRecord.from_dict(entry) for entry in payload]

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in self.records], handle, ensure_ascii=False, indent=2)

    def update_record(self, record: PoseRecord) -> None:
        for index, existing in enumerate(self.records):
            if existing.image == record.image:
                self.records[index] = record
                break
        else:
            self.records.append(record)

    def remove_record(self, image_name: str) -> None:
        self.records = [record for record in self.records if record.image != image_name]

    def find(self, image_name: str) -> Optional[PoseRecord]:
        for record in self.records:
            if record.image == image_name:
                return record
        return None

    def list_images(self) -> List[str]:
        return [record.image for record in self.records]


# ---------------------------------------------------------------------------
# VP-tree implementation closely follows the algorithm used in the JavaScript
# version.  The tree stores indices into the PoseDatabase to keep serialization
# compact while still supporting fast k-NN queries.
# ---------------------------------------------------------------------------


@dataclass
class _VPTreeNode:
    index: int
    threshold: float
    left: Optional["_VPTreeNode"]
    right: Optional["_VPTreeNode"]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Optional["_VPTreeNode"]:
        if payload is None:
            return None
        return cls(
            index=payload["index"],
            threshold=payload["threshold"],
            left=cls.from_dict(payload.get("left")),
            right=cls.from_dict(payload.get("right")),
        )


class VPTree:
    def __init__(
        self,
        items: Sequence[PoseRecord],
        distance_fn: Callable[[PoseRecord, PoseRecord], float],
        root: Optional[_VPTreeNode] = None,
    ) -> None:
        self.items = list(items)
        self.distance_fn = distance_fn
        self.root = root if root is not None else self._build(list(range(len(self.items))))

    def _build(self, indices: List[int]) -> Optional[_VPTreeNode]:
        if not indices:
            return None
        vantage_index = indices.pop(random.randrange(len(indices)))
        if not indices:
            return _VPTreeNode(vantage_index, 0.0, None, None)

        distances = [
            self.distance_fn(self.items[vantage_index], self.items[other_index])
            for other_index in indices
        ]
        median = float(np.median(distances)) if distances else 0.0
        left_indices = [idx for idx, distance in zip(indices, distances) if distance <= median]
        right_indices = [idx for idx, distance in zip(indices, distances) if distance > median]
        return _VPTreeNode(
            index=vantage_index,
            threshold=median,
            left=self._build(left_indices),
            right=self._build(right_indices),
        )

    def search(self, target: PoseRecord, k: int = 1) -> List[Tuple[float, PoseRecord]]:
        """Return the k nearest neighbors to the target pose."""

        if self.root is None or not self.items:
            return []

        heap: List[Tuple[float, int]] = []
        tau = float("inf")

        def _search(node: Optional[_VPTreeNode]) -> None:
            nonlocal tau
            if node is None:
                return

            vantage_record = self.items[node.index]
            dist = self.distance_fn(target, vantage_record)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, node.index))
                if len(heap) == k:
                    tau = -heap[0][0]
            elif dist < tau:
                heapq.heapreplace(heap, (-dist, node.index))
                tau = -heap[0][0]

            if node.left is None and node.right is None:
                return

            if dist < node.threshold:
                if dist - tau <= node.threshold:
                    _search(node.left)
                if dist + tau >= node.threshold:
                    _search(node.right)
            else:
                if dist + tau >= node.threshold:
                    _search(node.right)
                if dist - tau <= node.threshold:
                    _search(node.left)

        _search(self.root)
        neighbors = sorted(((-distance, self.items[index]) for distance, index in heap), key=lambda item: item[0])
        return neighbors

    def to_dict(self) -> dict:
        return {
            "size": len(self.items),
            "root": self.root.to_dict() if self.root else None,
        }

    def save(self, path: Path = PREBUILT_TREE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        items: Sequence[PoseRecord],
        distance_fn: Callable[[PoseRecord, PoseRecord], float],
        path: Path = PREBUILT_TREE_PATH,
    ) -> "VPTree":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VP-tree file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        root = _VPTreeNode.from_dict(payload.get("root"))
        return cls(items, distance_fn, root)


# ---------------------------------------------------------------------------
# Distance functions replicate the semantics of the original project while
# relying on Mediapipe-derived vectors.
# ---------------------------------------------------------------------------


def _expand_confidence(conf: Sequence[float], vector_length: int) -> np.ndarray:
    conf_array = np.asarray(conf, dtype=np.float32)
    if conf_array.size * 2 == vector_length:
        return np.repeat(conf_array, 2)
    return np.resize(conf_array, vector_length)


def weighted_distance(pose_a: PoseRecord, pose_b: PoseRecord) -> float:
    vector_a = np.asarray(pose_a.vector_xy, dtype=np.float32)
    vector_b = np.asarray(pose_b.vector_xy, dtype=np.float32)
    if vector_a.shape != vector_b.shape:
        raise ValueError("Pose vectors must have the same length for distance computation")

    conf_a = _expand_confidence(pose_a.vector_confidence, vector_a.size)
    conf_b = _expand_confidence(pose_b.vector_confidence, vector_b.size)
    weights = (conf_a + conf_b) / 2.0
    diff = vector_a - vector_b
    return math.sqrt(float(np.sum(weights * diff * diff)))


def cosine_distance(pose_a: PoseRecord, pose_b: PoseRecord) -> float:
    vector_a = np.asarray(pose_a.vector_xy, dtype=np.float32)
    vector_b = np.asarray(pose_b.vector_xy, dtype=np.float32)
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 1.0
    return 1.0 - float(np.dot(vector_a, vector_b) / denominator)


# ---------------------------------------------------------------------------
# Utility helpers for filesystem operations (listing, copying, deleting images)
# have been ported from the original Express services.
# ---------------------------------------------------------------------------


def list_images(debug: bool = False) -> List[str]:
    directory = DEBUG_IMAGE_DIR if debug else MIRROR_IMAGE_DIR
    return sorted(
        file.name
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def delete_image(image_name: str) -> None:
    target = MIRROR_IMAGE_DIR / image_name
    if target.exists():
        target.unlink()


# ---------------------------------------------------------------------------
# expand_image_database and rebuild_vptree are the two high-level entry points
# requested for automation.  Both functions are carefully documented to make it
# clear when to call each of them.
# ---------------------------------------------------------------------------


def expand_image_database(
    new_image_paths: Iterable[Path],
    estimator: Optional[PoseEstimator] = None,
    database: Optional[PoseDatabase] = None,
    image_directory: Path = MIRROR_IMAGE_DIR,
) -> List[PoseRecord]:
    """Copy new images, estimate their poses, and update the pose database.

    Args:
        new_image_paths: Iterable of filesystem paths to ingest.
        estimator: Optional PoseEstimator instance for dependency injection.
        database: Optional PoseDatabase instance (created on demand otherwise).
        image_directory: Destination directory for mirror images.

    Returns:
        A list of PoseRecord entries that were successfully added or updated.
    """

    created_estimator = False
    if estimator is None:
        estimator = PoseEstimator()
        estimator._ensure_initialized()
        created_estimator = True
    if database is None:
        database = PoseDatabase()

    processed_records: List[PoseRecord] = []
    image_directory.mkdir(parents=True, exist_ok=True)

    for path in new_image_paths:
        source = Path(path)
        if not source.exists() or source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        destination = image_directory / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(str(source), str(destination))

        record = estimator.estimate_from_file(destination)
        if record is None:
            destination.unlink(missing_ok=True)
            continue
        database.update_record(record)
        processed_records.append(record)

    database.save()
    if created_estimator:
        estimator.close()
    return processed_records


def rebuild_vptree(
    database: Optional[PoseDatabase] = None,
    tree_path: Path = PREBUILT_TREE_PATH,
    distance_function: Callable[[PoseRecord, PoseRecord], float] = weighted_distance,
) -> VPTree:
    """Rebuild the VP-tree from the current pose database."""

    if database is None:
        database = PoseDatabase()
    tree = VPTree(database.records, distance_function)
    tree.save(tree_path)
    return tree


# ---------------------------------------------------------------------------
# Public query helpers simplify matching logic.  They can be used in CLI mode
# or imported elsewhere.
# ---------------------------------------------------------------------------


def load_tree(database: Optional[PoseDatabase] = None, tree_path: Path = PREBUILT_TREE_PATH) -> VPTree:
    if database is None:
        database = PoseDatabase()
    return VPTree.load(database.records, weighted_distance, tree_path)


def find_nearest_matches(
    pose: PoseRecord,
    tree: VPTree,
    k: int = 1,
) -> List[Tuple[float, PoseRecord]]:
    return tree.search(pose, k)


# ---------------------------------------------------------------------------
# Real-time webcam support mirrors the original Move Mirror experience.  The
# helper below streams frames from a webcam, draws detected skeletons, and
# displays the closest matching mirror image from the database alongside the
# live feed.
# ---------------------------------------------------------------------------


def _resize_with_aspect(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or target_height <= 0:
        return np.zeros((max(target_height, 1), max(width, 1), 3), dtype=np.uint8)
    scale = target_height / float(height)
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (new_width, target_height))


def _create_match_panel(
    best_image: Optional[np.ndarray],
    matches: Sequence[Tuple[float, PoseRecord]],
    frame_height: int,
    panel_width: int = 360,
) -> np.ndarray:
    if best_image is not None:
        panel = _resize_with_aspect(best_image, frame_height)
    else:
        panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)

    if panel.shape[1] < panel_width:
        pad_width = panel_width - panel.shape[1]
        panel = cv2.copyMakeBorder(panel, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif panel.shape[1] > panel_width:
        panel = cv2.resize(panel, (panel_width, frame_height))

    if matches:
        text_lines = [
            f"{index}. {record.image} ({distance:.3f})"
            for index, (distance, record) in enumerate(matches, start=1)
        ]
        text_height = min(frame_height, 30 * len(text_lines) + 30)
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (panel.shape[1], text_height), (0, 0, 0), -1)
        panel = cv2.addWeighted(overlay, 0.6, panel, 0.4, 0)
        y = 25
        for line in text_lines:
            cv2.putText(
                panel,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 30

    return panel


def run_realtime_mirror(
    k: int = 1,
    camera_index: int = 0,
    database: Optional[PoseDatabase] = None,
    tree: Optional[VPTree] = None,
    distance_function: Callable[[PoseRecord, PoseRecord], float] = weighted_distance,
) -> None:
    """Stream webcam frames and display the closest mirror pose matches."""

    if database is None:
        database = PoseDatabase()
    if not database.records:
        raise RuntimeError("The pose database is empty. Add images before using the mirror mode.")

    if tree is None:
        try:
            tree = load_tree(database)
        except FileNotFoundError:
            tree = rebuild_vptree(database, distance_function=distance_function)

    with PoseEstimator(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as estimator:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open webcam at index {camera_index}")

        print("Press 'q' to quit the Move Mirror session.")

        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                estimate = estimator.estimate_from_array(frame, label="webcam", return_landmarks=True)
                pose_record: Optional[PoseRecord] = None
                landmarks = None
                if estimate is not None:
                    pose_record, landmarks = estimate
                if pose_record is None:
                    landmarks = None

                if landmarks is not None:
                    mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                matches: List[Tuple[float, PoseRecord]] = []
                if pose_record is not None:
                    matches = tree.search(pose_record, max(1, k))

                best_image = None
                if matches:
                    best_path = MIRROR_IMAGE_DIR / matches[0][1].image
                    best_image = cv2.imread(str(best_path))

                panel = _create_match_panel(best_image, matches[:k], frame.shape[0])

                status_text = (
                    f"Best match: {matches[0][1].image} ({matches[0][0]:.3f})"
                    if matches
                    else "Pose not detected"
                )
                color = (0, 255, 0) if matches else (0, 0, 255)
                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                combined = np.hstack((frame, panel))
                cv2.imshow("Move Mirror", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Command-line interface mirrors the developer tooling that previously lived in
# the Nuxt debugging pages.  It provides list/add/delete/build/match actions.
# ---------------------------------------------------------------------------


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move Mirror management toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available mirror images")
    list_parser.add_argument("--debug", action="store_true", help="List debugging images instead")

    add_parser = subparsers.add_parser("add", help="Add new mirror images and update the database")
    add_parser.add_argument("paths", nargs="+", type=Path, help="Image files to ingest")

    delete_parser = subparsers.add_parser("delete", help="Remove an image from the library")
    delete_parser.add_argument("image", type=str, help="Image filename to delete")

    subparsers.add_parser("build-tree", help="Rebuild the VP-tree from the pose database")

    match_parser = subparsers.add_parser("match", help="Find the closest matches for an image")
    match_parser.add_argument("image", type=Path, help="Image file to evaluate")
    match_parser.add_argument("-k", "--neighbors", type=int, default=1, help="Number of matches to return")

    mirror_parser = subparsers.add_parser("mirror", help="Run the real-time webcam Move Mirror")
    mirror_parser.add_argument("-k", "--neighbors", type=int, default=1, help="Number of matches to display")
    mirror_parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index to use")

    return parser.parse_args()


def _command_list(debug: bool) -> None:
    images = list_images(debug)
    for image in images:
        print(image)


def _command_add(paths: Sequence[Path]) -> None:
    records = expand_image_database(paths)
    print(f"Processed {len(records)} images")


def _command_delete(image: str) -> None:
    delete_image(image)
    database = PoseDatabase()
    database.remove_record(image)
    database.save()
    print(f"Deleted {image}")


def _command_build_tree() -> None:
    database = PoseDatabase()
    tree = rebuild_vptree(database)
    print(f"VP-tree built with {len(tree.items)} poses")


def _command_match(image_path: Path, neighbors: int) -> None:
    with PoseEstimator() as estimator:
        pose = estimator.estimate_from_file(image_path)
    if pose is None:
        print("Pose could not be estimated for the provided image")
        return
    database = PoseDatabase()
    tree = load_tree(database)
    matches = find_nearest_matches(pose, tree, neighbors)
    for distance, record in matches:
        print(f"{record.image}\t{distance:.6f}")


def _command_mirror(neighbors: int, camera_index: int) -> None:
    run_realtime_mirror(neighbors, camera_index)


# ---------------------------------------------------------------------------
# The main entry point wires the CLI commands together.  The script can be run
# directly or imported as a module.
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_arguments()
    if args.command == "list":
        _command_list(args.debug)
    elif args.command == "add":
        _command_add(args.paths)
    elif args.command == "delete":
        _command_delete(args.image)
    elif args.command == "build-tree":
        _command_build_tree()
    elif args.command == "match":
        _command_match(args.image, args.neighbors)
    elif args.command == "mirror":
        _command_mirror(args.neighbors, args.camera_index)


if __name__ == "__main__":
    main()
