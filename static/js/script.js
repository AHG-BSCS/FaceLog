document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const trainButton = document.getElementById('train');
    const video = document.getElementById('video');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');
    flashInterval = null;

    startButton.addEventListener('click', () => {
        if (startButton.textContent === 'Stop Camera') {
            startButton.textContent = 'Start Camera';
            registerButton.disabled = false;
            trainButton.disabled = false;
            video.src = "";
            fetch('/stop_feed');
        } else {
            startButton.textContent = 'Stop Camera';
            video.src = "/face_recognition";
            registerButton.disabled  = true;
            trainButton.disabled = true;
        }
    });

    registerButton.addEventListener('click', () => {
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
            video.src = "/face_capturing";
            startButton.disabled = true;
            trainButton.disabled = true;
            video.addEventListener('load', handleVideoLoad);
        }
    });

    trainButton.addEventListener('click', () => {
        trainButton.textContent = 'Training...';
        trainButton.disabled = true;
        startButton.disabled = true;
        registerButton.disabled = true;

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
                    trainButton.textContent = 'Train Model';
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
                }, 300);
                
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
