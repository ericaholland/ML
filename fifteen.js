/*
My code generate the boxes during the makeBoxes function and gives all boxes a class of .box and an id of
#box1 (or which ever tile it is). The background is starry night which is a horizontal photo, so I made the width
of it 600px so it would fill the square (the right part of the picture is cut off). I did not use the technique
of tracking the columns of the boxes, which I think is why I couldn't get the shuffle to work.

In class a few weeks ago you said it would be sufficient to just have everything work up until the shuffle
algorithm so that is where I left it.
 */

(function() {
    var emptyspot = [300, 300]; // the empty spot

    window.onload = function () {
        document.getElementById("shufflebutton").onclick = shuffle;
        makeBoxes();
        document.getElementById("box1").onmouseenter = highlight;

    }

    function makeBoxes() {
        // Makes boxes and initializes placement
        for (var i = 1; i < 16; i++) {
            var box = document.createElement("div");
            document.getElementById("puzzlearea").appendChild(box);
            box.classList.add("box");
            box.setAttribute("id", "box" + i);
            box.onmouseenter = highlight;
            box.onmouseleave = unhighlight;
            box.onmousedown = movetile;
            box.innerHTML = i;
            box.style.fontSize = "35pt";
            box.style.color = "white";
        }
    }

    function highlight() {
        //if mouse enters a highlightable box
        var that = this;
        if (movable(that) == true) {
            this.classList.add("hover");
        }
    }

    function unhighlight() {
        //to unhighlight when mouse leaves
        this.classList.remove("hover");
    }

    function movetile() {
        /* when tile is clicked, the movable function makes sure it is a movable tile
        then we swap the empty space with the clicked tile by moving the position of the clicked tile to the empty
        spot and reassigning the empty spot variable */
        var that = this;
        if (movable(that) == true) {
            var currentx = parseInt(window.getComputedStyle(that).getPropertyValue('left'));
            var currenty = parseInt(window.getComputedStyle(that).getPropertyValue('top'));
            var tempempty = [currentx, currenty];
            this.style.left = emptyspot[0] + "px";
            this.style.top = emptyspot[1] + "px";
            emptyspot = tempempty;
        }
    }

    function movable(that) {
        // checks to see if the tile that is hovered over/click can be moved
        var currentx = parseInt(window.getComputedStyle(that).getPropertyValue('left'));
        var currenty = parseInt(window.getComputedStyle(that).getPropertyValue('top'));
        var xdiff = emptyspot[0] - currentx; // distance between the clicked tile and empty spot
        var ydiff = emptyspot[1] - currenty;
        if (xdiff == 100 && ydiff == 0) {
            //empty to the right: if the clicked tile is to the left of the empty
            return true;
        }
        else if (xdiff == 0 && ydiff == 100) {
            //empty above: if the clicked tile is below the empty
            return true;
        }
        else if (xdiff == -100 && ydiff == 0) {
            //empty to the left: if the clicked tile is to the right of the empty
            return true;
        }
        else if (xdiff == 0 && ydiff == -100) {
            // empty below: if the clicked tile is above the empty
            return true;
        }
        else {
            // if the tile isn't one of those four options, it isn't moveable
            return false;
        }
    }

    function shuffle() {
        alert("Sorry, the shuffle button doesn't work!");
    }
} ());
