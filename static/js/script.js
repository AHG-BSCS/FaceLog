document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const trainButton = document.getElementById('train');
    const analyzeButton = document.getElementById('analyze');
    const video = document.getElementById('video');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');
    // cameraIndex = document.getElementById('cameraSelect').value;
    flashInterval = null;

    fetch('/load_models')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                startButton.disabled = true;
                analyzeButton.disabled = true;
                alert(data.error);
            }
        });
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
            analyzeButton.disabled = false;
            video.src = null;
            fetch('/stop_feed');
        } else {
            fetch('/load_models')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                        return;
                    }
                });
            startButton.textContent = 'Stop Camera';
            // video.src = `/face_recognition?cameraIndex=${cameraIndex}`;
            video.src = "/face_recognition";
            registerButton.disabled  = true;
            trainButton.disabled = true;
            analyzeButton.disabled = true;
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
            video.src = null;
            startButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.disabled = false;
            fetch('/stop_feed');
            video.removeEventListener('load', handleVideoLoad);
        } else {
            registerButton.textContent = 'Stop Capturing';
            // video.src = `/face_capturing?cameraIndex=${cameraIndex}`;
            video.src = "/face_capturing";
            startButton.disabled = true;
            trainButton.disabled = true;
            analyzeButton.disabled = true;
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
        analyzeButton.disabled = true;
        trainButton.textContent = 'Training...';
        fetch('/training')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    startButton.disabled = false;
                    analyzeButton.disabled = false;
                    alert(data.message);
                }
            });
        trainButton.disabled = false;
        registerButton.disabled = false;
        trainButton.textContent = 'Update Model';
    });

    analyzeButton.addEventListener('click', () => {
        analyzeModel();
    });

    function analyzeModel() {
        fetch('/load_models')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
            });
        startButton.disabled = true;
        registerButton.disabled = true;
        trainButton.disabled = true;
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';
        video.src = '/analyze_model';
        video.addEventListener('load', handleVisualizerLoad);
    }

    function handleVideoLoad() {
        video.removeEventListener('load', handleVideoLoad);
        const userName = prompt("Enter Name:");
        if (!userName) {
            alert("User name is required.");
            registerButton.textContent = 'Register Now';
            video.src = null;
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
                    video.src = null;
                    fetch('/stop_feed');
                    return;
                }
            }, 800);
        }
    }

    function handleVisualizerLoad() {
        video.removeEventListener('load', handleVisualizerLoad);
        startButton.disabled = false;
        registerButton.disabled = false;
        trainButton.disabled = false;
        analyzeButton.disabled = false;
        analyzeButton.textContent = 'Analyze Model';
    }
});