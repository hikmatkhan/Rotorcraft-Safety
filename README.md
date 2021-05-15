# Rotorcraft-Safety
Repository for doing analytics and inference on the open source flight safety dataset.

# Setup
Create a virtual environment and install dependencies using `pipenv` (assumes Python3.8+):
```shell script
python3.8 -m pipenv install .
```

# Running

`python -m run.py` in the terminal or through an IDE should download the flight safety data from the URL,
unzip the file, and load it into a Pandas dataframe. If you want to change where the data is downloaded to locally, just change some of the arguments in the functions.
