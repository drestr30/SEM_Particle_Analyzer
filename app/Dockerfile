FROM python:3.9-slim

#COPY ./ /app
WORKDIR /code
COPY requirements.txt /code
COPY classification.py /code

RUN apt update
#install nano editor in image
RUN apt-get install nano
#install opencv linux dependences
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libglib2.0-0

RUN pip install --upgrade pip
#RUN pip install --no-cache-dir torch==1.9.1
RUN pip install -r /code/requirements.txt

#WORKDIR /app
ENV PYTHONPATH=/code

EXPOSE 8051

RUN python classification.py
CMD ["streamlit", "run", "/code/demo.py"]
