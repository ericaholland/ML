<!DOCTYPE html>
<html>
<head>
    <title>PHP2</title>
    <meta charset="utf-8" />

    <?php
        $items =  glob("student/*.jpg"); /* gives list of food photos */
        for ($i = 0; $i < count($items); $i++) {
            /* change file names into just the food name (cut off the .jpg part) */
        }
        $items = [apple, carrot]; /* just hardcoded this in to work on it */
    ?>

</head>
<body>
    <form action="EricaHollandPHP3.php" method="post">
        <div>
            Food item:
            <select name="food".
                <? for ($i = 0; $i < count($items); $i++) { ?>
                <option> <?= $items[$i] ?></option>
                <? } ?>
            </select>
        </div>
        <div>Quantity: <input type="text" name="quantity" size="2" /> </div>
        <div> <input type="submit" value="Order" /> </div>
    </form>
</body>
</html>
