var express = require("express");
var socket = require("socket.io");
var WebSocket = require('ws');

// App Setup
var app = express();
var server = app.listen("8000", function(){
  console.log("listening to requests on port 8000");
});
var ws = new WebSocket.Server({ port: 8001 });
// Static files
app.use(express.static("public"));

var str = -1;

// Socket Setup
var io = socket(server);

io.on("connection",function(socket){
  console.log("made socket connection", socket.id);
  io.sockets.emit("This is a test from Server")
  socket.on("chat", function(data,ws){
    if(data == undefined){
        console.log("data from phone undefined")
    }else{
    io.sockets.emit("chat",data);
    //send to andriod here instead of console prints
    }
    console.log(data);
    console.log(data["handle"], data["message"]);
    str = data[9]
    if((str == "0") || (str == "1")){
      console.log("From phone we got" + str);
      //ws.emit(str)
    }
  });
});


// ws server

var strs = 1;
ws.on('connection', function connection(ws) {
  console.log("made socket connection with FitBit");
  ws.send("Hi from Server")
  ws.on('message', function incoming(message) {
    //if(message.length > 20){
    //  var readings = [];
    //  while (message.length > 20){
        //console.log('received: %s',message.slice(33,35)); //27:35
    //    readings.push(message.slice(33,35));
    //    message = message.slice(37);
    //  }
      //console.log('received: ', readings);
    //}else{
      console.log('received: %s', message);
    //}

    setInterval(function(){
      if (str == "1" || str == "0"){
        if(str == "1"){
          str = "reject"
        }else{
          str = "accept"
        }
        ws.send(String(str));
        console.log("We sent this decision to the watch: " + str);
      }
    },10000);
  });
});
