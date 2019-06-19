"use strict";

var queue = [ ];

var mouseDown = 0;

var background_active = 1;
var multiBrush = 1;

var xm, ym;
var xb, yb;
var dx, dy;

var doc = document;
var body = doc.body;
var win = window;
var canvas = body.children[0];
var ctx = canvas.getContext("2d");
var width ;
var height ;
var lastX = width / 2;
var lastY = height / 2;
var math = Math;

onWindowResize();

var backgroundColor='#00bfff' ;

function fillTextMultiLine(ctx, text, x, y) {
  var lineHeight = ctx.measureText("M").width * 1.2;
  var lines = text.split("\n");
  for (var i = 0; i < lines.length; ++i) {
    ctx.fillText(lines[i], x, y);
    y += lineHeight;
  }
}

ctx.fillStyle = backgroundColor ;
ctx.fillRect(0, 0, width, height);
ctx.fillStyle = 'red' ;
ctx.font = "30px Arial";
ctx.textAlign = "center";
ctx.fillText(`Drag your mouse to draw`,width/2, height/2);
ctx.fillText(`Press S to download a snapshot`,width/2, 30+height/2);
const helptext = 
      `Drag your mouse to draw
Press S to download a snapshot`
//fillTextMultiLine(ctx,helptext,width/2,height/2) ;

doc.onmousemove = mousemove;
doc.onmouseup = function() {
  mouseDown = 0;
};

doc.onmousedown = function(event) {
  mouseDown = 1;
  mousemove(event);
};

doc.onkeydown = function(event) {
  var key = event.keyCode;
  if (key == 83) saveImage() ;
  if (key == 32) background_active = !background_active;
  if (key == 77) multiBrush = !multiBrush;
};

	window.addEventListener("resize", onWindowResize, false);

function onWindowResize(event) {
  width = (canvas.width = win.innerWidth);
  height = (canvas.height = win.innerHeight);
}


//setInterval(zoom, 33);

let request; // in case we want to cancel
let frame = 0 ;
animate();
function animate() {
  request = requestAnimationFrame(animate);
  if (frame++%2){
    zoom() ;  
  }
}


function draw(x, y, radius) {
  ctx.strokeStyle = "black";
  ctx.lineWidth = radius / 7;
  ctx.beginPath();

  if (mouseDown) {
    ctx.arc(x, y, radius, 0, math.PI * 2, 1);
  }

  ctx.closePath();

  ctx.fillStyle = "white";
  ctx.stroke();
  ctx.fill();
}

function mousemove(event) {
  const x = event.clientX - canvas.offsetLeft;
  const y = event.clientY - canvas.offsetTop;

  const dx = lastX - x;
  const dy = lastY - y;
  queue.push({ x: x, y: y, r: (math.sqrt(dx * dx + dy * dy) + 1) | 0 });

  lastX = x;
  lastY = y;
}

function zoom() {
  if (background_active) {
    {
      ctx.save();
      ctx.translate(width / 2, height / 2);
      ctx.scale(1.03, 1.03);
      ctx.translate(-width / 2, -height / 2);
      ctx.drawImage(canvas, 0, 0);

      ctx.globalAlpha = 0.01 ;
      ctx.fillStyle = backgroundColor ;
      ctx.fillRect(0, 0, width, height);
      ctx.restore();
    }
  }

  if (queue.length == 0 && mouseDown) {
    queue.push({ x: lastX, y: lastY, r: 1 });
  }

  while (queue.length) {
    const state = queue.shift() ;
    {
      if (multiBrush) {
        xm = state.x % 200;
        ym = state.y % 200;
        for (xb = -200; xb <= width; xb += 200) {
          for (yb = -200; yb <= height; yb += 200) {
            draw(xb + xm, yb + ym, state.r);
          }
        }
      } else {
        draw(state.x, state.y, state.r);
      }
    }
  }
}

function saveImage() {
	// see https://stackoverflow.com/a/45789588/242848
	const link = document.createElement("a");
	link.setAttribute("download", "contrail-doodler.png");
	link.setAttribute(
		"href",
		canvas.toDataURL("image/png").replace("image/png", "image/octet-stream")
	);
	link.click();
}
