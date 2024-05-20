document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const startButton = document.getElementById('start');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const faceNameElement = document.createElement('div');
    const registerButton = document.getElementById('register');

    startButton.textContent = 'Start Webcam';

    startButton.addEventListener('click', () => {
        if (startButton.textContent === 'Stop Webcam') {
            video.src = "";
            fetch('/stop_feed');
            startButton.textContent = 'Start Webcam';
        } else {
            video.src = "/video_feed";
            startButton.textContent = 'Stop Webcam';
        }
    });

    registerButton.addEventListener('click', () => {
        const name = prompt("Enter your name:");
        if (name) {
            $.ajax({
                url: '/register',
                type: 'POST',
                data: { name: name },
                success: (response) => {
                    alert(response.message);
                },
                error: (error) => {
                    console.error("Error registering user: ", error);
                }
            });
        }
    });
});
