document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const trainButton = document.getElementById('train');
    const analyzeButton = document.getElementById('analyze');
    const attendanceButton = document.getElementById('attendance');
    const video = document.getElementById('video');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');

    // for attendance values
    const currentDate = new Date();
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
    const month = monthNames[currentDate.getMonth()];
    const day = String(currentDate.getDate());
    const year = String(currentDate.getFullYear());
    const formattedDate = `${month} ${day} ${year}`;

    document.querySelector('.card-attendance').textContent = `Attendance for ${formattedDate}`;
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
        if (startButton.style.backgroundColor === 'maroon') {
            startButton.style.backgroundColor = '#6a3acb';
            startButton.style.backgroundImage = "url('static/image/recognize-start.png')";
            // startButton.src = '../image/recognize-start.png';
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
            startButton.style.backgroundColor = 'maroon';
            startButton.style.backgroundImage = "url('static/image/recognize-stop.png')";
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
        if (registerButton.style.backgroundColor === 'maroon') {
            clearInterval(flashInterval);
            registerButton.style.backgroundImage = "url('static/image/register.png')";
            registerButton.backgroundColor = '#6a3acb';
            imageCount.textContent = '';
            video.src = null;
            startButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.disabled = false;
            fetch('/stop_feed');
            video.removeEventListener('load', handleVideoLoad);
        } else {
            registerButton.style.backgroundColor = 'maroon';
            registerButton.style.backgroundImage = "url('static/image/stop.png')";
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
        trainButton.style.backgroundImage = "url('static/image/training.png')";
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
        trainButton.style.backgroundImage = "url('static/image/train.png')";
    });

    analyzeButton.addEventListener('click', () => {
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
        analyzeButton.style.backgroundImage = "url('static/image/training.png')";
        video.src = '/analyze_model';
        video.addEventListener('load', handleVisualizerLoad);
    });

    attendanceButton.addEventListener('click', () => {
        const tableBody = document.getElementById('logTable').getElementsByTagName('tbody')[0];
        const table = document.getElementById('attendance-table')
        
        if (attendanceButton.style.backgroundColor === 'maroon') {
            table.style.visibility = 'collapse';
            attendanceButton.style.backgroundImage = "url('static/image/attendance-view.png')";
            attendanceButton.style.backgroundColor = '#6a3acb';
            tableBody.innerHTML = '';  // Clear any existing rows
        }
        else {
            fetch('/read_attendance')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                alert(data.error);
                return;
                }
                else {
                table.style.visibility = 'visible';
                attendanceButton.style.backgroundImage = "url('static/image/attendance-close.png')";
                attendanceButton.style.backgroundColor = 'maroon';
                tableBody.innerHTML = '';  // Clear any existing rows
                i = 1;
                
                data.forEach(row => {
                    const newRow = tableBody.insertRow();
                    newRow.insertCell(0).textContent = i;
                    newRow.insertCell(1).textContent = row.Name;
                    newRow.insertCell(2).textContent = row.Time;
                    newRow.insertCell(3).textContent = row.Probability
                    i++;
                });
                }
            })
            .catch(error => alert(error));
        }
    });

    function handleVideoLoad() {
        video.removeEventListener('load', handleVideoLoad);
        const userName = prompt("Enter Name:");
        if (!userName) {
            alert("User name is required.");
            registerButton.style.backgroundColor = '#6a3acb';
            registerButton.style.backgroundImage = "url('static/image/register.png')";
            startButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.disabled = false;
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
                    registerButton.style.backgroundImage = "url('static/image/register.png')";
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
        analyzeButton.style.backgroundImage = "url('static/image/analyze.png')";
    }
});