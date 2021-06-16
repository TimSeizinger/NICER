FROM python:3

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run a test if the container has all dependencies
RUN python -c "import imagenet_c, matplotlib, numpy, cv2, pandas, pickleshare, PIL, pytorch_lightning, rawpy, skimage, scipy, tensorboard, tensorboard_plugin_wit, torch, torchvision, tifffile, tqdm, future, jinja2"