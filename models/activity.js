var mongoose = require('mongoose');

var Schema = mongoose.Schema;

var ActivitySchema = new Schema(
  {
    name: { type: String }, 
    _id: { type: String }
  }
);

//Export model
module.exports = mongoose.model('Activity', ActivitySchema);

