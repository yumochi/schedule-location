var mongoose = require('mongoose');

var Schema = mongoose.Schema;

var ActivityListSchema = new Schema(
  {
    name: { type: String }, 
    content: [{type: String, ref: 'Activity'}]
  }
);


//Export model
module.exports = mongoose.model('ActivityList', ActivityListSchema);

