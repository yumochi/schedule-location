var express = require('express');
var router = express.Router();

var schedule_controller = require('../controllers/scheduleController');

/* GET users listing. */
router.get('/', schedule_controller.retrieve_json);


module.exports = router;
