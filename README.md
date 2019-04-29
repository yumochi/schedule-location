# schedule-location

Welcome to Schedule Locations, a very basic UI to showcase result of location classification on construction schedules, using a combination of Pytorch, Express, Pug, MongoDB, and Node. The application meant to take a csv version of construction schedule and classify each word in relation to location level. A RNN is used for the classification and is capable of much more than illustrated here. This is meant to be a simple demonstration of combining js and python to create a GUI to work with neural nets. The original inspiration of the front end and database is based on the MDN tutorial for the local library app.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

node, npm 

https://nodejs.org/en/download/

python package

These are listed in requirements.txt and can be installed with 

```
pip install -r ./public/python_scripts/requirements.txt
```

pyTorch

https://pytorch.org/

### Installing

Clone repo

```
git clone git@github.com:yumochi/schedule-location.git
cd schedule-location
```

Install js libraries and middlewares needed are listed in the package.json and can be installed with 

```
npm install
```

create assets and input folders under public

```
mkdir ./public/assets/
mkdir ./public/input/
```

setup MLab to store and retrieve information

go through the walkthrough to set up a simple database

https://mlab.com/

change the mongo variable to work with your database on line 14 in app.js

```
var mongoDB = 'mongodb://userName:userPassword@mlabDatabaseURL'; 
```
Note that the userName and userPassword refer to the database you setup on mlab and not your mlab username and password itself.

run application on address in browser

https://localhost:3000/

## Running the tests

### Store Schedule CSV on to MLab

This will show your that the application is sending info into your MLab databse

- Copy a schedule of your choice into public/assets/

- Go to 

- https://localhost:3000/

- Upload the schedule with the file input

- Submit the file 

The application should redirect you to a new page where the information is displayed on an input table

Go to your MLab database to see if the if is uploaded

### Run Python Script to Produce Results

This will show you that the app is successfully retrieving info from MLab and running the scripts

- Click the RUN button on the page to process the uploaded schedule

The application should again redirect you to a new page where each of the word in the activity description is color coded in term of their importance in relation to level in the schedule. 

## Deployment

Currently this project is still being improved.

## Built With

* [Express](https://expressjs.com/) - The web framework used
* [MLab](https://mlab.com/) - Database used
* [PUG](https://pugjs.org/api/getting-started.html) - Used to generate HTML
* [PyTorch](https://pytorch.org/) - Used to create neural networks

## Contributing

Feel free to work with our current code to explore and learn.

## Authors

* **Yumo Chi** - *Initial work* - (https://github.com/yumochi)

* **Wilfredo Torres** - *Initial work* - (https://github.com/wtorresc)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to MDN for the tutorial and many other stackoverflow examples we tried.
