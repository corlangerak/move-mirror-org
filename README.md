# Move Mirror (Mediapipe Edition)

This repository now provides a single Python script, `move_mirror.py`, that
covers the full feature set of the original Nuxt.js Move Mirror clone while
switching pose estimation from PoseNet to Mediapipe.

## Features
- Estimate human poses for static images using Mediapipe Pose.
- Maintain a persistent pose database (`data/posedata.json`) compatible with the
  original project structure.
- Build and serialize a VP-tree (`data/prebuilt_vptree.json`) for fast nearest
  neighbour queries.
- Manage the mirror image library by adding or removing images directly from the
  command line.

## Requirements
- Python 3.9+
- [Mediapipe](https://developers.google.com/mediapipe)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

Install the dependencies with:

```bash
pip install mediapipe opencv-python numpy
```

## Usage

All functionality is exposed through `move_mirror.py`:

```bash
python move_mirror.py list            # List mirror images
python move_mirror.py add path/*.jpg  # Add new images and update the database
python move_mirror.py delete image.jpg
python move_mirror.py build-tree      # Rebuild the VP-tree
python move_mirror.py match query.jpg --neighbors 3
```

The script expects mirror pose images to live in `assets/images/mirror-poses`
and optional debugging images in `assets/images/debug`.

## Extensibility

Two key functions are available for automation:
- `expand_image_database(...)` copies new images, estimates their poses, and
  updates the pose database.
- `rebuild_vptree(...)` rebuilds and persists the VP-tree whenever the pose
  database changes.

Both functions can be imported from `move_mirror.py` and reused in other
applications.
