# YOLOv5 FastAPI Web

This repository hosts a FastAPI web application allowing users to upload images for inference using a pre-trained YOLOv5 model, receiving results in JSON format.

![YOLOv5 FastAPI Web](https://user-images.githubusercontent.com/47000850/171301696-fe31b6fd-a2c4-4b2c-9029-f11ce1ddfb64.png)

## Installation

Ensure Python 3.10 or later is installed, along with dependencies listed in `requirements.txt`, including `torch>=1.7`. You can install dependencies using:

```
pip install -r requirements.txt
```

## Example Usage

### Minimal FastAPI Example

Navigate to the `client_server_example` folder for a minimal client/server wrapper of YOLOv5 with FastAPI and HTML forms.

1. Run the server with:
   ```
   python server_minimal.py
   ```
   or 
   ```
   uvicorn server_minimal:app --reload
   ```

2. Test the server with:
   ```
   python client_minimal.py
   ```
   or by navigating to `localhost:8000` in your web browser and selecting "Try It Out" in the POST request section.

### Inference Methods

1. Initialize the server with:
   ```
   python server.py
   ```
   (use `--help` for other arguments) or
   ```
   uvicorn server:app --reload
   ```

2. Test the server:
   - Use `client.py` to upload images and receive JSON inference results.
   - Open `localhost:8000` in your web browser to use the web form for image upload and model selection.
   - Open `http://localhost:8000/drag_and_drop_detect` for a drag and drop interface.

Models are automatically downloaded on first use and cached on disk.

![City Street Results](https://user-images.githubusercontent.com/47000850/171300877-e3941e01-1aa0-4816-9cf9-6947481b4ec8.png)

## API Documentation

Auto-generated API documentation is available at `localhost:8000/docs`. The API route provides JSON inference results.

## Developer Notes

### `server.py`

Contains FastAPI server code and helper functions.

### Jinja2 Frontend Templates (`/templates` folder)

| File | Description |
| --- | --- | 
| `layout.html` | Base template with navbar common to all pages. `home.html` and `drag_and_drop_detect.html` extend this template. |
| `home.html` | Basic web form for uploading images, model selection, and inference. Server renders bbox image and returns results via `templates/show_results.html`. |
| `drag_and_drop_detect.html` | Implements a Drag & Drop interface for image upload. Sends image and parameters to server's `/detect` endpoint, rendering image + bboxes in the browser. Box labels are raised above the outline to prevent overlap. |

## Credits

This repository is a wrapper around YOLOv5 from Ultralytics: [GitHub - ultralytics/yolov5](https://github.com/ultralytics/yolov5)

Results_to_json function is modified from: [GitHub Gist](https://gist.github.com/decent-engineer-decent-datascientist/81e04ad86e102eb083416e28150aa2a1)
