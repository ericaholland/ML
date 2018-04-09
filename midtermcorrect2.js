// currently highlight based on the div game (so they don't individually highlight

window.onload = function() {
    document.getElementById("add").onclick = addWord;
   // document.getElementById("game").onmouseenter = addColor;
   // document.getElementById("game").onmouseleave = removeColor;
   // document.getElementById("game").onmouseup = deleteDiv;
}

function addWord(){

    var word = document.getElementById("input").value;
    var p = document.createElement("p");
    p.innerHTML = word;
    p.onmouseenter = addColor;
    p.onmouseleave = removeColor;
    p.setAttribute("id", "list");
    document.getElementById("game").appendChild(p);
}

function addColor() {
    this.style.backgroundColor = "cyan";
    var size = window.getComputedStyle(this).getPropertyValue('font-size');
    size = parseInt(size) + 2 + "px";
    this.style.fontSize = size;
}

function removeColor() {
    this.style.backgroundColor = "white";
    var size = window.getComputedStyle(this).getPropertyValue('font-size');
    size = parseInt(size) - 2 + "px";
    this.style.fontSize = size;
}

function deleteDiv(){
    var parent = document.getElementById("game");
    var child = document.getElementById("list");
    parent.style.fontSize = "18px"; //change font size back down
    parent.removeChild(child);
}
