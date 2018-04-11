<!DOCTYPE html>
<html>
<head>
    <title>PHP1</title>

    <meta charset="utf-8" />

    <?php
        $number = $_GET["number"];
        $number = $number % 2;
        ?>
</head>

<body>
    <div>
        <p> <?
            if ($number == 1){ ?>
            Your number is odd.
            <? }
            else { ?>
            Your number is even.
             <? } ?>
        </p>
    </div>

</body>
</html>
