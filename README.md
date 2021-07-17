# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The only necessary library to run the code beyond the Anaconda distribution of Python is the Plotly library.  
The code should run with no issues using Python versions 3.*.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

This project aims to train software engineering and data engineering skills, combining machine learning techniques, pipelines in addition to front and back end.

The goal of this project is to have a web app where an emergency worker can input a new message and get classification results in several categories from the message data provided by Figure Eight.


## File Descriptions <a name="files"></a>

1. In the data folder is the .py pipeline that generates a sqlite database from the csv files
    - disaster_messages.csv -> file with the messages used in model classification
    - disaster_categories.csv -> file with the categories of the messages
    - process_data.py -> pipeline that merge, clean and save data into a sqlite database
    - DisasterResponse.db -> sqlite database with messages and categories
2. In the models folder is the .py pipeline which generates a picke file of the created model
    - train_classifier.py -> ml pipeline that fits a model on messages and classification data
    - classifier.pkl -> pickle file with model
3. In the app folder is the back and front end application
    - templates -> folder with the html files of web app
    - run.py -> back end of the web app


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data belongs to [Udacity](https://www.udacity.com/). Feel free to use the code here as you would like! 

== End ==