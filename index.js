const path = require('path');
const uuid = require('uuid/v4');
const express = require('express');
const bodyParser = require('body-parser');
const spawn = require("child_process").spawn;
const EventEmitter = require('events')

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.use((req, res, next) => { req.id = uuid(); next(); });

const supportedLanguage = ['it', 'en'];

// Strategy 1: spin a python process at each request passing input parameters
// Too much slow. There's too much overhead in starting the process and
// loading the model.
function predict(res, lang, toPredict) {
    const pathToScript = path.resolve(__dirname, './predict.py');
    const pythonProcess = spawn('python', [pathToScript, lang, toPredict]);
    pythonProcess.stdout.on('data', (data) => {
        console.log(data.toString());
        res.end(data.toString().trim());
    });
}

// Strategy 2: start the python process once and communicates with it through stdin and stdout
const py2 = spawn('python', [path.resolve(__dirname, './predict2.py')]);
py2.stdin.setDefaultEncoding('utf-8');
py2.stdout.on('data', (data) => console.log(data.toString().trim())); // log result
py2.stderr.on('data', (data) => console.log(data.toString())); // log errors

app.get('/', (req, res) => {
    res.sendFile(path.resolve(__dirname, './index.html'));
});

app.post('/classify/:lang', (req, res) => {
    const lang = req.params.lang;
    if (!supportedLanguage.includes(lang)) {
        res.status(404);
        res.end();
        return
    }
    let toPredict = req.body.review;
    toPredict = toPredict.replace(/\r?\n|\r/g, ' ');
    if (toPredict.length === 0) {
        res.status(200);
        res.end('und');
        return;
    }
    if (toPredict.length > 1000) {
        res.status(400);
        res.json({ error: 'Review length must be below 1000 characters'});
        return;
    }

    function callback(result) {
        splits = result.toString().trim().split(' ');
        // In case of multiple asynchronous calls,
        // check if the response is the expected one for this request
        // if it is publish a response and remove this callback function from
        // the listeners.
        if (splits[0] === req.id) {
            res.end(splits[1].trim());
            py2.stdout.off('data', callback);
        }
    }
    // Register listener to the python process stdout
    py2.stdout.on('data', callback);

    console.log(`${req.id} ${lang} ${toPredict}`); // log request
    // Writes data to the python process stdin
    py2.stdin.write(`${req.id} ${lang} ${toPredict}\n`);

    // Too SLOW!
    // predict(res, lang, toPredict);
});

app.listen(port, () => {
    console.log(`Listening on port ${port}`);
})
