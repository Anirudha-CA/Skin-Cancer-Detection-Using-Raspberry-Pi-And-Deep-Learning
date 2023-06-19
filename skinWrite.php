<?php
//error_reporting(E_ALL & ~E_WARNING & ~E_NOTICE);


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

else { echo "Connected to mysql database. "; }

		
	if (count($_POST['text']))
		{
			$text = $_POST['text'];

			

// Update your tablename here
	$sql = "INSERT INTO {$tbname} (text) VALUES ('".$text."')";
 


		if ($conn->query($sql) === TRUE) 
		{
		    echo "Values inserted in MySQL database table.";
		} 
		else 
		{
		    echo "Error: " . $sql . "<br>" . $conn->error;
		}
		}
			
	

// Close MySQL connection
$conn->close();



?>
