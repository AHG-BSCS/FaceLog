document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');
    flashInterval = null;

    startButton.textContent = 'Start Camera';
    registerButton.textContent = 'Register Now';

    startButton.addEventListener('click', () => {
        if (startButton.textContent === 'Stop Camera') {
            startButton.textContent = 'Start Camera';
            video.src = "";
            fetch('/stop_feed');
        } else if (registerButton.textContent === 'Stop Capturing') {
            alert("Busy capturing faces!");
        } else {
            startButton.textContent = 'Stop Camera';
            video.src = "/face_recognition";
        }
    });

    registerButton.addEventListener('click', () => {
        if (registerButton.textContent === 'Stop Capturing') {
            clearInterval(flashInterval);
            registerButton.textContent = 'Register Now';
            video.src = "";
            fetch('/stop_feed');
            video.removeEventListener('load', handleVideoLoad);
        } else if (startButton.textContent === 'Stop Camera') {
            alert("Busy recognizing faces!");
        } else {
            registerButton.textContent = 'Stop Capturing';
            video.src = "/face_capturing";
            // Add event listener for when the video image is loaded
            video.addEventListener('load', handleVideoLoad);
        }
    });

    function handleVideoLoad() {
        const userName = prompt("Enter Name:");
        if (!userName) {
            alert("User name is required.");
            registerButton.textContent = 'Register Now';
            video.src = "";
            fetch('/stop_feed');
            return;
        } else {
            fetch('/capture_images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_name: userName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    alert(data.message);
                }
            });
        
            // Flash effect every 0.8 seconds for 5 times
            let count = 0;
            flashInterval = setInterval(() => {
                flash.style.opacity = 1;
                setTimeout(() => {
                    flash.style.opacity = 0;
                }, 300); // Flash duration
                
                count++;
                imageCount.textContent = `${count}`;
                if (count > 50) {
                    clearInterval(flashInterval);
                    registerButton.textContent = 'Register Now';
                    imageCount.textContent = '';
                    video.src = "";
                    fetch('/stop_feed');
                    return;
                }
            }, 800);
        }
        video.removeEventListener('load', handleVideoLoad);
    }
});
