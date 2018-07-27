FROM nvcr.io/nvidia/tensorflow:18.06-py3

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir tensorflow==1.8.0 numpy==1.14.0 matplotlib Augmentor keras==2.1.6 opencv-python pillow bokeh scipy==1.0.0

