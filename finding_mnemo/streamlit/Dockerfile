FROM python:3.9

WORKDIR /code

COPY ./streamlit_requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./search_app.py /code/search_app.py
  
CMD ["streamlit", "run", "search_app.py"]
