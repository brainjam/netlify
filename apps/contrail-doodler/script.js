"use strictx";

const queue = [ ];

let mouseDown = 0;

let background_active = 1;
let multiBrush = 1;

let xm, ym;
let xb, yb;
let dx, dy;

const canvas = document.getElementById('c');
const ctx = canvas.getContext("2d");
let width;
let height;
let lastX ;
let lastY ;

onWindowResize();

const backgroundColor = "#00bfff";

ctx.fillStyle = backgroundColor;
ctx.fillRect(0, 0, width, height);

document.onmousemove = mousemove;
document.onmouseup = function() {
  mouseDown = 0;
};

document.onmousedown = function(event) {
  mouseDown = 1;
  mousemove(event);
};

document.onkeydown = function(event) {
  var key = event.keyCode;
  if (key == 83) saveImage();
  if (key == 32) background_active = !background_active;
  if (key == 77) multiBrush = !multiBrush;
};

window.addEventListener("resize", onWindowResize, false);

function onWindowResize(event) {
  let temp ;
  if (width){
    temp = ctx.getImageData(0,0,width,height) ;  
  }
  width = canvas.width = window.innerWidth;
  height = canvas.height = window.innerHeight;
  //console.log({width,height})
  if (temp){
    ctx.putImageData(temp,0,0) ;
  }
}

let request; // in case we want to cancel
let frame = 0;
animate();
function animate() {
  request = requestAnimationFrame(animate);
  frame++ ;
  zoom();
}

function draw(x, y, radius) {
  ctx.save() ;
  ctx.strokeStyle = "black";
  ctx.lineWidth = radius / 7;
  ctx.beginPath();

  if (mouseDown) {
    ctx.arc(x, y, radius, 0, Math.PI * 2, 1);
  }

  ctx.closePath();

  ctx.fillStyle = "white";
  ctx.stroke();
  ctx.fill();
  ctx.restore() ;
}

function mousemove(event) {
  const x = event.clientX - canvas.offsetLeft;
  const y = event.clientY - canvas.offsetTop;

  const dx = lastX - x;
  const dy = lastY - y;
  queue.push({ x: x, y: y, r: (Math.sqrt(dx * dx + dy * dy) + 1) | 0 });

  lastX = x;
  lastY = y;
}

function zoom() {
  if (background_active) {
    {
      ctx.save();
      const xbias = Math.random() ;
      const ybias = Math.random() ;
      ctx.translate(xbias + width / 2, ybias + height / 2);
      ctx.scale(1.01, 1.01);
      ctx.translate(-width / 2, -height / 2);
      ctx.drawImage(canvas, 0, 0);
      ctx.globalAlpha = frame%32 ? .005 : .04;
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);
      ctx.restore();
    }
  }

  if (queue.length == 0 && mouseDown) {
    queue.push({ x: lastX, y: lastY, r: 1 });
  }

  while (queue.length) {
    const state = queue.shift();
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
  zoom() ;
  link.setAttribute("download", "contrail-doodler.png");
  link.setAttribute(
    "href",
    canvas.toDataURL("image/png").replace("image/png", "image/octet-stream")
  );
  link.click();
}