function showSpinner(show) {
    const spinner = document.querySelector('.loader'); // Use .loader for consistency with your CSS
    spinner.style.display = show ? 'inline-block' : 'none'; // Adjusted to 'inline-block' for visibility
}

document.getElementById('uploadBtn').addEventListener('click', function() {
    document.getElementById('videoInput').click(); // Trigger file input click
});

document.getElementById('videoInput').addEventListener('change', function(event) {
    if (event.target.files.length > 0) {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file); // Ensure this matches the name expected by the Flask route
        //file now in the object FormData

        // Hide the upload button and show the spinner
        document.getElementById('uploadBtn').style.display = 'none';
        showSpinner(true);

        // Perform the upload
        fetch('/upload', { //file sent to server using Fetch API
            method: 'POST',
            body: formData,
        })// the following is code for handling the server response
        .then(response => {
            if (response.ok) { // if upload and processing succesful, the server responds with a JSON object that includes the URL of the processed video
                return response.json(); //was text, changed to json response.  server sends back a JSON object that includes the URL for the processed video, not just a filename.// Assuming the server responds with the filename
            } 
            throw new Error('Upload failed'); //will show if the server encoutners an issue processing the file (but still responds!)
            //^^ok status of false 
        })
        .then(data => { 
            // Hide the spinner and show the processed video
            showSpinner(false);
            const processedVideo = document.getElementById('processedVideo');
            processedVideo.src = data.processed_video_url; // Update this path if necessary // src = `/uploads/${filename}`;// data.processed_video_url  is the URL provided by the server. ensures that the client correctly requests the processed video from wherever the server says it's located (we dont manually say it anymore.)This lcoation should now be the /processed/ directory
            //^^updating the 'src' attribute of the processedVideo element with JSON URL of processed video to make it visible
            processedVideo.style.display = 'block';
            document.querySelector('.css-selector').classList.add('black-background'); //making background black when processed video is displayed

             // The 'once: true' option auto-removes the event listener after it fires once
        })
        .catch(error => { //will catch error that occure DURING the FETCH operation
            console.error('Error:', error);
            alert('Failed to upload video.');
            // Optionally reset UI on failure
            document.getElementById('uploadBtn').style.display = 'inline-block';
            showSpinner(false);
        });
    }
});

function requestFullscreen(element) {
    if (element.requestFullscreen) {
      element.requestFullscreen();
    } else if (element.mozRequestFullScreen) { /* Firefox */
      element.mozRequestFullScreen();
    } else if (element.webkitRequestFullscreen) { /* Chrome, Safari & Opera */
      element.webkitRequestFullscreen();
    } else if (element.msRequestFullscreen) { /* IE/Edge */
      element.msRequestFullscreen();
    }
  }



