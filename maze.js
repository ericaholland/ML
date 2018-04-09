var wallHit = false;

window.onload = function() {
    document.getElementById("boundary1").onmouseover = overBoundary;
    var boundaries = document.querySelectorAll("div#maze div.boundary");
    for (var i = 0; i < boundaries.length; i++) {
        boundaries[i].onmouseover = overBoundary;
    }
    var end = document.getElementById("end");
    end.onmouseover = winner;
    var start = document.getElementById("start");
    start.onclick = reset;
    document.body.onmousemove = overBody;
};

function overBoundary() {
    // turn the wall you hit red
    var section = document.getElementById("boundary1");
        if( !section.classList.contains("invalid")) {
            section.classList.add("youlose");
        }
    var areas = document.querySelectorAll("div#maze div.boundary");
        for (var i = 0; i < areas.length; i++){
            if(!areas[i].classList.contains("invalid")){
                areas[i].classList.add("youlose");
            }
        }
    wallHit = true;
    winner;
}

// test for mouse being over document.body so that the player
// can't cheat by going outside the maze (haxor exercise)
function overBody(event) {
    if (wallHit === false && event.target == document.body) {
        overBoundary(event);
    }
}

function winner() {
    var status = document.getElementById("status");
    if( wallHit === false){
        status.innerHTML = "You win";
    }
    else{
        status.innerHTML = "You lose";
    }
}

function reset() {
    wallHit = false;
    document.getElementById("status").innerHTML = "Find the end";
    var boundaries = document.querySelectorAll("div#maze div.boundary");
    for (var i = 0; i < boundaries.length; i++){
        boundaries[i].classList.remove("youlose");
    }
}