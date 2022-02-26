FROM 3.8.12-booster

WORKDIR app

COPY models/age_model_pretrained.h5 age_model_pretrained.h5
COPY models/gender_model_pretrained.h5 gender_model_pretrained.h5
COPY models/emotion_model_pretrained.h5 emotion_model_pretrained.h5
COPY models/haarcascade_frontface_default.xml haarcascade_frontface_default.xml
COPY requirements.txt requirements.txt
COPY utils.py utils.py
COPY ./github ./github
COPY ./face_detector_web ./face_detector_web
COPY ./pages ./pages
COPY app.py app.py
COPY Makefile Makefile
COPY multipage.py multipage.py
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
