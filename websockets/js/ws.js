var ndarray = require('ndarray');
var ws = new WebSocket("ws://localhost:6011/websocket");
ws.onopen = function() {
    metadata = {
        name: "array",
        shape: [3,3],
        dtype:"float32"
    };
    ws.send(JSON.stringify(metadata));
    array = ndarray(new Float32Array([1,0,0,0,1,0,0,0,1]), metadata.shape);
    ws.send(array.data.buffer);


};
ws.onmessage = function (evt) {
    console.log(evt.data);
};
