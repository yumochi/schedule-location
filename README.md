# schedule-location

Welcome to Schedule Locations, a very basic UI to showcase result of location classification on construction schedules, using a combination of Pytorch, Express, Pug, MongoDB, and Node. The application meant to take a csv version of construction schedule and classify each word in relation to location level. A RNN is used for the classification and is capable of much more than illustrated here. This is meant to be a simple demonstration of combining js and python to create a GUI to work with neural nets.

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

setup mlab to store and retrieve information

go through the walkthrough to set up a simple database

https://mlab.com/

change the mongo variable to work with your database on line 14 in app.js

```
var mongoDB = 'mongodb://userName:userPassword@mlabDatabaseURL'; 
```
Note that the userName and userPassword refer to the database you setup on mlab and not your mlab username and password itself.

run application on address in browser

http://localhost:3000

## Running the tests

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
