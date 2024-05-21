document.addEventListener('DOMContentLoaded', async () => {
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const trainButton = document.getElementById('train');
    const video = document.getElementById('video');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');
    const cameraSelect = document.getElementById('cameraSelect');
    let currentStream = null;
    let flashInterval = null;

    async function getCameras() {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
    }

    async function startCamera() {
        const deviceId = cameraSelect.value;
        const constraints = {
            video: {
                deviceId: { exact: deviceId }
            }
        };
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
    }

    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        video.srcObject = null;
    }

    cameraSelect.addEventListener('change', () => {
        if (startButton.textContent === 'Stop Camera') {
            stopCamera();
            startCamera();
        }
    });

    startButton.addEventListener('click', async () => {
        if (startButton.textContent === 'Stop Camera') {
            startButton.textContent = 'Start Camera';
            registerButton.disabled = false;
            trainButton.disabled = false;
            stopCamera();
            fetch('/stop_feed');
        } else {
            startButton.textContent = 'Stop Camera';
            await startCamera();
            registerButton.disabled = true;
            trainButton.disabled = true;
        }
    });

    registerButton.addEventListener('click', () => {
        if (registerButton.textContent === 'Stop Capturing') {
            clearInterval(flashInterval);
            registerButton.textContent = 'Register Now';
            imageCount.textContent = '';
            stopCamera();
            startCamera();
            startButton.disabled = false;
            trainButton.disabled = false;
            fetch('/stop_feed');
            video.removeEventListener('loadeddata', handleVideoLoad);
        } else {
            registerButton.textContent = 'Stop Capturing';
            startCamera().then(() => {
                video.srcObject = currentStream;
                startButton.disabled = true;
                trainButton.disabled = true;
                video.addEventListener('loadeddata', handleVideoLoad);
            });
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
            stopCamera();
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
                    stopCamera();
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
        video.removeEventListener('loadeddata', handleVideoLoad);
    }

    await getCameras();
});
