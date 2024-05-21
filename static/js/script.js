document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const trainButton = document.getElementById('train');
    const video = document.getElementById('video');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');
    // cameraIndex = document.getElementById('cameraSelect').value;
    flashInterval = null;

    // fetch('/list_cameras')
    //     .then(response => response.json())
    //     .then(cameras => {
    //         const cameraGroup = document.getElementById('cameraGroup');
    //         cameras.forEach((camera, index) => {
    //             const option = document.createElement('option');
    //             option.value = camera;
    //             option.textContent = `Camera ${index + 1}`;
    //             cameraGroup.appendChild(option);
    //         });
    //     });

    startButton.addEventListener('click', () => {
        startButton.disabled = true;
        if (startButton.textContent === 'Stop Camera') {
            startButton.textContent = 'Start Camera';
            registerButton.disabled = false;
            trainButton.disabled = false;
            video.src = "";
            fetch('/stop_feed');
        } else {
            startButton.textContent = 'Stop Camera';
            // video.src = `/face_recognition?cameraIndex=${cameraIndex}`;
            video.src = "/face_recognition";
            registerButton.disabled  = true;
            trainButton.disabled = true;
        }
        // Enable button after 3 seconds to prevent multiple clicks
        setTimeout(() => {
            startButton.disabled = false;
        }, 3000);
    });

    registerButton.addEventListener('click', () => {
        registerButton.disabled = true;
        if (registerButton.textContent === 'Stop Capturing') {
            clearInterval(flashInterval);
            registerButton.textContent = 'Register Now';
            imageCount.textContent = '';
            video.src = "";
            startButton.disabled = false;
            trainButton.disabled = false;
            fetch('/stop_feed');
            video.removeEventListener('load', handleVideoLoad);
        } else {
            registerButton.textContent = 'Stop Capturing';
            // video.src = `/face_capturing?cameraIndex=${cameraIndex}`;
            video.src = "/face_capturing";
            startButton.disabled = true;
            trainButton.disabled = true;
            video.addEventListener('load', handleVideoLoad);
        }
        // Enable button after 3 seconds to prevent multiple clicks
        setTimeout(() => {
            registerButton.disabled = false;
        }, 3000);
    });

    trainButton.addEventListener('click', () => {
        trainButton.disabled = true;
        startButton.disabled = true;
        registerButton.disabled = true;
        trainButton.textContent = 'Training...';

        alert('Waiting for training to complete...');
        fetch('/training')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    trainButton.disabled = false;
                    startButton.disabled = false;
                    registerButton.disabled = false;
                    trainButton.textContent = 'Update Model';
                    alert(data.message);
                }
            });
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
        
            // Flash effect
            let count = 0;
            flashInterval = setInterval(() => {
                flash.style.opacity = 1;
                setTimeout(() => {
                    flash.style.opacity = 0;
                }, 100);
                
                count++;
                imageCount.textContent = `${count}/50`;
                if (count > 50) {
                    clearInterval(flashInterval);
                    registerButton.textContent = 'Register Now';
                    imageCount.textContent = '';
                    video.src = "";
                    fetch('/stop_feed');
                    fetch('/train_model')
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                alert(data.error);
                            } else {
                                alert(data.message);
                            }
                        });
                    return;
                }
            }, 800);
        }
        video.removeEventListener('load', handleVideoLoad);
    }
});