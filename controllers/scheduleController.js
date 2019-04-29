const fs = require('fs');

const multer  = require('multer')
const upload = multer({ dest: 'uploads/' })

const { body,validationResult } = require('express-validator/check');
const { sanitizeBody } = require('express-validator/filter');

const Activity = require('../models/activity');

const ActivityList = require('../models/activitylist');

/**
 * @param {obj} schedule in obj form
 * @return {obj} cleaned schedule
 */
function clean(obj) {
  for (var propName in obj) { 
    if (obj[propName] === null || obj[propName] === undefined) {
      delete obj[propName];
    }
  }
}

/**
 * @param {dateString} string representing number 
 * @return {Date}
 */
function getJsDateFromExcel(rawText) {

      // JavaScript dates can be constructed by passing milliseconds
      // since the Unix epoch (January 1, 1970) example: new Date(12312512312);

      // 1. Subtract number of days between Jan 1, 1900 and Jan 1, 1970, plus 1 (Google "excel leap year bug")             
      // 2. Convert to milliseconds.
      let excelDate = Number(rawText);
      return new Date((excelDate - (25567 + 1))*86400*1000);

}

/**
 * @param {rawText} uploaded csv file representing construction schedules in raw string form 
 * @return {activityList} array of activity name of all construction activities in schedule
 * @return {parsedSchedule} array of objects containing the activity of schedule in object form
 */
function parseSchedule(rawText) {

        let parsedSchedule = [];

        let allTextLines = rawText.split('\n');
        allTextLines = allTextLines.slice(2);
        const headersLength = 7; // allTextLines[0].split(',');

        for (let col_i of allTextLines) {
              // split content based on comma
              const data = col_i.split(',');
              const headers = data.slice(0, headersLength);
              if (headers.length) {
                const result = {
                  activityId: headers[0],
                  activityStatus: headers[1],
                  wbsCode: headers[2],
                  wbsName: headers[3],
                  activityName: headers[4],
                  startDate: getJsDateFromExcel(headers[5]),
                  endDate: getJsDateFromExcel(headers[6]),
                };
                const activity = {
                  activityName: headers[4]
                }
                parsedSchedule.push(result);
              }
        }

        return parsedSchedule;

}

exports.upload_schedule_get = function(req, res, next){
    res.render('index', { title: 'Map Schedule Locations'});
}

exports.upload_schedule_post = function(req, res, next){
  
    fs.readFile(req.file.path, (err, data) => { 
        if (err) throw err; 
  
        const rawText = data.toString();
        const schedule = parseSchedule(rawText);

        let activityNames = []

        for (let scheduleActivity of schedule){

            activityNames.push(scheduleActivity.activityName);
            // Create a genre object with escaped and trimmed data.
            const activity = new Activity(
              { 
                name: scheduleActivity.activityName,
                _id: scheduleActivity.activityId //This is required, or a new ID will be assigned!
              }
            );
            // Data from form is valid.

            // Extract the validation errors from a request.
            const errors = validationResult(req);

       
            if (!errors.isEmpty()) {
              // There are errors. Render the form again with sanitized values/error messages.
              res.render('index', { title: 'Upload Schedule', errors: errors.array()});
            return;
            }
            else {
              // Data from form is valid.
              // Check if Genre with same name already exists.
              Activity.findOne({ 'name': scheduleActivity.activityName })
                .exec( function(err, found_activity) {
                   if (err) { return next(err); }

                   if (found_activity) {
                     // Genre exists, redirect to its detail page.
                     console.log('duplicate activity')
                   }
                   else {

                     activity.save(function (err) {
                       if (err) { return next(err); }
                     });

                   }

                 });
            }
        }

        const activityList = new ActivityList(
            {
                content: activityNames,
                name: req.file.originalname
            }
        );

        // Data from form is valid.
        // Check if Genre with same name already exists.
        ActivityList.findOne({ 'name': req.file.originalname })
          .exec( function(err, found_activity_list) {
             if (err) { return next(err); }

             if (found_activity_list) {
               // Genre exists, redirect to its detail page.
               // res.redirect(found_activity_list);
               console.log('duplicate schedule')
             }
             else {

               activityList.save(function (err) {
                 if (err) { return next(err); }
                 // Genre saved. Redirect to genre detail page.
                 res.render('schedule_display', { title: 'Display Uploaded Schedule', schedule: {name: req.file.originalname, activities:schedule } } );
               });

             }

           });
    })
}

// Display retrieved json.
exports.retrieve_json = function(req, res, next) { 
      

    body('name', 'Schedule name required')
    console.log(req.query.scheduleName)

    // Check if Activity with name exists.
    ActivityList.findOne({ "name": req.query.scheduleName })
      .exec( function(err, found_activity_list) {
         if (err) { return next(err); }

         if (found_activity_list) {
            // Use child_process.spawn method from  
            // child_process module and assign it

                let content = found_activity_list.content
                clean(content)
                const path = "./public/input/rawinput.json"
                      fs.writeFile(path, content, function(err) {
                    if (err) {
                        res.render('error');
                    }
                    else{
                        var exec = require("child_process").exec;
                
                        exec(`python ./public/python_scripts/generate.py`, (error, stdout, stderr) => {
                          if (error) {
                            throw error
                            return;
                          }
                            stdout = stdout.replace(/\'/g, '"');
                            // stdout = stdout.substring(1, stdout.length-1);
                            let predictions = []

                            const commaSplitData = stdout.split(',');

                            for (let i in commaSplitData){
                                if (content[i] && commaSplitData[i])
                                {
                                  let encodedActivity = []
                                  const activityName = content[i].split(' ');
                                  const activityPrediction = commaSplitData[i].split(' ');
                                  for (let j in activityPrediction){
                                    const word = activityName[j];
                                    const prediction = activityPrediction[j];
                                    encodedActivity.push({word: word, locLevel: prediction});
                                  }
                                  predictions.push(encodedActivity)
                                }
                            }
                            res.render('schedule_result', { title: 'Display Processed Schedule', schedule: predictions } );
   
                        });
                    }
                });

         }
         else {

           res.render('error');

         }

       });


};