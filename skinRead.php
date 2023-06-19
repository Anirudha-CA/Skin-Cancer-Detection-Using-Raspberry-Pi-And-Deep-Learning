<?php



    $host = "sg2nlmysql15plsk.secureserver.net:3306";		         // host = localhost because database hosted on the same server where PHP files are hosted
    $dbname = "iotdb";              // Database name
    $username = "iotroot";		// Database username
    $password = "iot@123";	        // Database password
	$tbname = "skin";         // Database table name

// Establish connection to MySQL database
$conn = new mysqli($host, $username, $password, $dbname);


// Check if connection established successfully
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

else { echo "Connected to mysql database.\n"; }

 
	$sql = "SELECT * FROM {$tbname} ORDER BY id DESC LIMIT 1;";  // Update your tablename here
	//$sql = "SELECT from_SW FROM {$tbname} ORDER BY id DESC LIMIT 1;";

$result = $conn->query($sql);

if ($result->num_rows > 0) {

    // output data of each row
    while($row = $result->fetch_assoc()) {
		  echo "txt: " . $row["text"]."";
    
}
} else {
    echo "0 results";
}




// Close MySQL connection
$conn->close();



?>