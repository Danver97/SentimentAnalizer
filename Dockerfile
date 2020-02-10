FROM nikolaik/python-nodejs:latest

COPY . /src

WORKDIR /src

RUN npm install

RUN ./install_python_dependencies.sh

RUN python training.py

EXPOSE 3000

CMD npm start