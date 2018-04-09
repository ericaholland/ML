// This prints a fibonacci code up to 20
// in terminal, I installed php [man php], then typed [php -S localhost:8000]
// (the 8000 is kind of irrelevant what number it is)
// will give you a doc route.
// http://localhost:8000/WebstormProjects/NetDev/ClassPractice/hello.php
// ^ that is the url to put it. Its basically the path after EricaHolland because that's the root

<html>
<body>
<ul>

<?php
$first = 0;
$second = 1;

for ($i = 0; $i < 20; $i++){ ?>

<li> <?= $first ?> </li>

<?php
$temp=$second;
$second+=$first;
$first=$temp;
} ?>

</ul>
