FROM python:3.6
RUN apt -y update && apt -y install libgl1-mesa-glx
WORKDIR /usr/app
RUN mkdir videos

ADD ./requirements.txt ./
RUN pip install -r requirements.txt
ADD ./ ./

CMD ["python3", "main.py"]
