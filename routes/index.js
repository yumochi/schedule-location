const express = require('express');
const router = express.Router();

const multer = require("multer")
const upload = multer({ dest: "../public/assets/" })

var schedule_controller = require('../controllers/scheduleController');

/* GET users listing. */
router.get('/', schedule_controller.upload_schedule_get);

router.route("/")
    /* replace foo-bar with your form field-name */
    .post(upload.single("schedule-file"), schedule_controller.upload_schedule_post)

module.exports = router;
