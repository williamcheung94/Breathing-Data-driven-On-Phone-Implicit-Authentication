// make connection
//var socket = io.connect("192.168.1.104:8000");
var socket = io.connect("localhost:8000");

// Query DOM
var message = document.getElementById("message");
var handle = document.getElementById("handle");
var btn = document.getElementById("send");
var output = document.getElementById("output");

// Emit events
btn.addEventListener("click", function(){
  socket.emit("chat", {
    message:message.value,
    handle:handle.value
  });
});

// Listen for addEventLinstener
socket.on("chat",function(data){
  output.innerHTML +="<p><strong>" + data.handle + ":</strong>" + data.message + "</p>";
});
