## Simple python docker dev example for the official docker docs
https://docs.docker.com/language/python/containerize/

 docker build -t langchain-llm-movie .
 docker images
 docker run -d -p 8000:8000 langchain-llm-movie

 ### to start FASTAPI server 
 fastapi dev app.py

 # virtual env

 python3 -m venv .venv
 #  activate by
  source .venv/bin/activate

  # test it 
  which python

  # Add .gitignore
  echo "*" > .venv/.gitignore
# to install from Install from requirements.txt

   pip install -r requirements.txt

# deactivate virtual env 
deactivate



